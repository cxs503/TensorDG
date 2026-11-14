"""Tests for CASE-DG-ADVECT-ORDER plumbing and the observed order of accuracy.

The heavy reference comparison lives in
``benchmarks/case_dg_advect_order.py`` (cluster run, ~5 min).  Here we keep the
cheap, CI-sized checks:

* the extracted reference table is present, self-consistent and carries the
  provenance hashes the report cites;
* the benchmark's own order estimator behaves (recovers a known slope);
* the TensorDG ladder really is order ``p + 1`` at ``p = 1, 2`` on coarse
  meshes (rate tolerance 0.2 here, the strict 0.10 gate belongs to the full
  benchmark run).
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import torch

from benchmarks.case_dg_advect_order import fit_with_floor, fitted_order
from tensordg.advection_1d import l2_error, solve_advection

REF = pathlib.Path("validation/advection_order/ref_table.json")


def test_reference_table_is_present_and_well_formed():
    payload = json.loads(REF.read_text())
    prov = payload["provenance"]
    assert prov["extractor"] == "tools/fetch_ref_advorder.py"
    assert len(prov["eprint_sha256"]) == 64
    assert len(prov["table_block_sha256"]) == 64
    assert len(payload["columns"]) == 4
    rows = payload["rows"]
    assert [r["N"] for r in rows] == [20, 40, 80, 160, 320]
    for row in rows:
        for column in payload["columns"]:
            value = row[column["name"]]["l2"]
            assert 0.0 < value < 1.0
    # errors must decrease monotonically on every column
    for column in payload["columns"]:
        vals = [r[column["name"]]["l2"] for r in rows]
        assert all(a > b for a, b in zip(vals[:-1], vals[1:])), column["name"]


def test_fitted_order_recovers_a_known_slope():
    cells = [20, 40, 80, 160]
    errors = [3.0 * (1.0 / n) ** 2.5 for n in cells]
    assert abs(fitted_order(cells, errors) - 2.5) < 1e-12
    # points at the round-off floor are excluded, the rest still fit
    floored = errors + [1e-16]
    order, nfit, dropped = fit_with_floor(cells + [320], floored)
    assert nfit == 4 and dropped == [320]
    assert abs(order - 2.5) < 1e-12


def _ladder(degree, cells, cfl=0.02, final_time=0.25):
    out = []
    for n in cells:
        centers, coeff, _ = solve_advection(
            ncells=n, degree=degree, final_time=final_time, cfl=cfl)
        out.append(l2_error(centers, coeff, final_time=final_time))
    return out


def test_ladder_reproduces_p_plus_one_order():
    torch.set_num_threads(2)
    for degree, expected in ((1, 2.0), (2, 3.0)):
        errors = _ladder(degree, (8, 16, 32))
        rate = fitted_order((8, 16, 32), errors)
        assert math.isfinite(rate)
        assert abs(rate - expected) < 0.2, (degree, errors, rate)

def test_time_order_probe_reports_design_order_with_correct_sign():
    """Regression for the round-2 instrument defect (M5 reported "not_run").

    The fixed-mesh probe refines the *step count*; handing ``dt`` to the shared
    estimator (which assumes an increasing resolution variable) negated the
    slope and made every degree look wrong.  Both the sign and the magnitude
    are asserted here, so the defect cannot silently return.
    """
    from benchmarks.case_dg_advect_order_matched_rk import dt_order_probe

    torch.set_num_threads(2)
    probe = dt_order_probe(2, levels=4)
    measured = probe["fitted_time_order"]
    assert measured is not None, probe
    assert measured > 0.0, ("sign flipped", measured)
    assert abs(measured - probe["design_order"]) < 0.2, (measured, probe)


def test_matched_rung_uses_reference_stage_count():
    """Degree ``p`` runs with ``p + 1`` stages, i.e. the reference's column."""
    from tensordg.runge_kutta import tableau_for_degree

    for degree in (1, 2, 3, 4):
        tab = tableau_for_degree(degree)
        assert tab.stages == degree + 1, (degree, tab.name, tab.stages)
