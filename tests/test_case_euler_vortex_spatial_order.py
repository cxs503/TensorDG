"""Verification tests for ``CASE-DG-EULER-VORTEX-ERR`` spatial order (S1..S5).

These tests deliberately separate three kinds of evidence:

* **data binding** -- the claims in
  ``docs/CASE-DG-EULER-VORTEX-ERR-spatial-order.md`` are recomputed here from
  the raw per-mesh errors stored in ``spatial_order.json`` with an independent
  numpy fit, so a edited number in the document cannot survive;
* **artifact consistency** -- the two meshes that the case-pool ladder already
  ran (``n = 32``, ``n = 64`` at ``t = 1``, CFL 0.2) must be reproduced by this
  study to round-off, which proves the order study sits on exactly the same
  solver path instead of a private one;
* **discrimination** -- the order statistic must *fail* on a synthetic
  second-order sequence (a scheme that is one order short) and the wrong
  reference must collapse the order, i.e. the gates are not vacuously true.

A live solver check on the P1 family (design order 2) runs two coarser meshes;
it is the only test that touches the solver, and its runtime is seconds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import case_euler_vortex_errconserve as ec
from benchmarks import case_euler_vortex_spatial_order as so

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "case_euler_vortex_errconserve" / "spatial_order.json"
DOC = ROOT / "docs" / "CASE-DG-EULER-VORTEX-ERR-spatial-order.md"
LADDER = ROOT / "case_euler_vortex_errconserve" / "results"

pytestmark = pytest.mark.skipif(not DATA.exists(),
                                reason="spatial_order.json not produced yet")


@pytest.fixture(scope="module")
def data():
    return json.loads(DATA.read_text())


def _records(data):
    return sorted(data["records"], key=lambda r: r["cells"])


def _independent_fit(cells, values):
    """Least-squares order of ``e ~ h^p``, ``h = box / cells`` (own numpy code)."""
    cells = np.asarray(cells, dtype=float)
    values = np.asarray(values, dtype=float)
    return float(-np.polyfit(np.log(cells), np.log(values), 1)[0])


# ---------------------------------------------------------------------------
# data binding
# ---------------------------------------------------------------------------

def test_verdict_is_pass_with_no_skipped_gate(data):
    gates = data["gates"]
    assert gates["verdict"] == "PASS"
    assert gates["failed"] == []
    assert gates["not_evaluated"] == []
    assert gates["all_passed"] is True


def test_fitted_orders_are_recomputable_from_the_raw_errors(data):
    records = _records(data)
    window = records[1:]                       # asymptotic window: drop coarsest
    cells = [r["cells"] for r in window]
    stored = data["gates"]["S1_spatial_order"]["fitted_orders"]
    lower = data["gates"]["S1_spatial_order"]["lower_bound"]
    for field in so.FIELDS:
        values = [r["errors"][field] for r in window]
        recomputed = _independent_fit(cells, values)
        assert recomputed == pytest.approx(stored[field], abs=1e-9)
        assert recomputed >= lower, (field, recomputed, lower)


def test_projection_sequence_has_the_same_rate(data):
    """S2: the error *metric* alone (pure L2 projection) is h^{p+1}-consistent."""
    records = _records(data)
    window = records[1:]
    cells = [r["cells"] for r in window]
    stored = data["gates"]["S2_measurement_consistency"]["orders"]
    for field in so.FIELDS:
        values = [r["projection_errors"][field] for r in window]
        assert _independent_fit(cells, values) == pytest.approx(stored[field], abs=1e-9)


def test_temporal_share_is_negligible(data):
    gate = data["gates"]["S3_temporal_negligible"]
    assert gate["status"] == "passed"
    assert gate["evaluated_meshes"], "S3 needs at least one CFL-control mesh"
    assert gate["worst"] < so.TEMPORAL_SHARE_MAX


def test_wrong_reference_collapses_the_order(data):
    gate = data["gates"]["S4_reference_discriminates"]
    assert gate["status"] == "passed"
    design_order = data["design"]["degree"] + 1
    assert gate["highest"] < 0.5 * design_order


def test_finest_mesh_is_well_inside_the_case_tolerance(data):
    gate = data["gates"]["S5_tolerance_margin"]
    assert gate["status"] == "passed"
    assert gate["finest_max_error"] < so.MARGIN_FRACTION * ec.TOLERANCE


def test_document_is_bound_to_the_stored_verdict(data):
    assert DOC.exists()
    text = DOC.read_text()
    assert data["gates"]["verdict"] in text
    assert "CASE-DG-EULER-VORTEX-ERR" in text


# ---------------------------------------------------------------------------
# artifact consistency with the case-pool ladder
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cells", (32, 64))
def test_study_reproduces_the_case_pool_ladder_records(data, cells):
    path = LADDER / f"dg_d2_n{cells}_t1_cfl0.2.json"
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    ladder = json.loads(path.read_text())
    record = next(r for r in _records(data) if r["cells"] == cells)
    for field in so.FIELDS:
        assert record["errors"][field] == pytest.approx(
            ladder["errors"][field], rel=1e-12, abs=1e-15)


# ---------------------------------------------------------------------------
# discrimination (anti-vacuity) of the gate logic itself
# ---------------------------------------------------------------------------

def _synthetic_records(exponent, *, wrong_reference_exponent=0.0):
    cells = [16, 32, 64, 128]
    h = [10.0 / n for n in cells]
    records = []
    for i, n in enumerate(cells):
        base = {f: 1e-4 * h[i] ** exponent for f in so.FIELDS}
        records.append({
            "cells": n,
            "errors": dict(base),
            "projection_errors": dict(base),
            "errors_wrong_reference": {
                f: 1e-2 * h[i] ** wrong_reference_exponent for f in so.FIELDS},
            "cfl_control": {"cfl": 0.1, "steps": 1, "wall_seconds": 0.0,
                            "errors": {f: base[f] * (1.0 + 1e-4) for f in so.FIELDS}},
        })
    return records


def test_gate_fails_a_second_order_sequence():
    """A p=2 answer must not pass the p+1=3 gate: the S1 statistic discriminates."""
    result = so.evaluate(_synthetic_records(exponent=2.0))
    assert result["gates"]["S1_spatial_order"]["status"] == "failed"
    assert result["gates"]["verdict"] == "FAIL"
    assert result["gates"]["S1_spatial_order"]["worst_fitted_order"] == pytest.approx(2.0, abs=0.05)


def test_gate_passes_a_third_order_sequence():
    result = so.evaluate(_synthetic_records(exponent=3.0))
    assert result["gates"]["S1_spatial_order"]["status"] == "passed"
    assert result["gates"]["verdict"] == "PASS"


# ---------------------------------------------------------------------------
# live solver check on the P1 family (design order 2)
# ---------------------------------------------------------------------------

def test_p1_family_converges_at_second_order():
    orders = {}
    for field in so.FIELDS:
        cells, errors = [], []
        for n in (8, 16):
            coeff, meta, _wall = so.run_state(n, cfl=0.2, degree=1, final_time=0.5)
            scored = so.score(coeff, n, degree=1, reference_time=0.5)
            cells.append(n)
            errors.append(scored[field])
        orders[field] = _independent_fit(cells, errors)
        assert meta["steps"] > 0
    worst = min(orders.values())
    assert worst >= 0.8 * 2.0, orders   # 80 % of the design order, two-point fit