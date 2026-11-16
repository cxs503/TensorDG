"""Tests for the round-3 stage/order-cap case
(:mod:`benchmarks.case_dg_advect_order_stage_cap`).

Two things are guarded here:

* the *barrier audit* -- the table the matched criterion consults must be the
  inversion of the published minimal-stage counts, and every registered tableau
  must respect it (a tableau ordered above its stage count would make the
  criterion self-contradictory);
* the *measurement itself* -- the ``p = 5`` / ``p = 6`` rungs must really show
  the spatial orders ``p + 1`` the DG operator promises, and the time-error
  share at the case's CFL must stay inside the 10 % window, so the claim "the
  measured rate is the spatial rate" cannot rot unnoticed.

The ladders used here stop at two meshes to keep the suite fast; the full
three-mesh ladder and the CFL scan live in the benchmark's stored artifacts.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from benchmarks.case_dg_advect_order_stage_cap import (CFL, DOC, OUT,
                                                       barrier_audit,
                                                       cfl_budget, rung,
                                                       time_split)
from tensordg.runge_kutta import (MAX_ORDER_FOR_STAGES, TABLEAUS,
                                  max_order_for_stages, tableau_for_degree)


def test_barrier_audit_matches_published_and_covers_registered_tableaus():
    audit = barrier_audit()
    assert audit["table_matches_published"] is True
    assert audit["table"] == audit["expected"]
    assert audit["all_tableaus_within_barrier"] is True
    assert audit["refused_above_certified_range"], "18 stages must be refused"


def test_tableau_selection_at_and_beyond_the_cap():
    """``p = 5`` is still reachable (6 stages); ``p = 6`` is the documented cap.

    ``p = 5`` needs six stages, which is exactly Butcher's 6-stage fifth-order
    scheme -- order and stage count agree again.  ``p = 6`` would need seven
    stages; none is registered, so *the same* six-stage tableau is used and the
    time order (5) drops below the spatial order (7) that the rung certifies.
    """
    assert tableau_for_degree(5).name == "rk5"
    assert tableau_for_degree(5).stages == 6
    assert tableau_for_degree(5).order == max_order_for_stages(6) == 5
    assert tableau_for_degree(6).name == "rk5"
    assert tableau_for_degree(6).stages == 6          # 7 would be needed
    assert tableau_for_degree(6).order == 5           # documented cap
    assert max_order_for_stages(7) == 6               # 7 stages *could* do 6


@pytest.mark.parametrize("degree,floor", [(5, 5.8), (6, 6.8)])
def test_rung_self_convergence_reproduces_p_plus_one(degree, floor):
    """The extended rung converges at the spatial order ``p + 1`` (no ref table)."""
    r = rung(degree, cells=(6, 12))
    assert r["integrator"] == "rk5"
    assert r["expected_spatial_order"] == degree + 1
    slopes = r["pairwise_orders"]
    assert len(slopes) == 1
    assert slopes[0] > floor, (degree, slopes, r["l2"])
    assert all(np.isfinite(r["l2"]))
    # the ladder is strictly decreasing and the errors stay above round-off
    assert r["l2"][0] > r["l2"][1] > 1e-13


def test_time_error_share_stays_inside_the_window_at_the_case_cfl():
    """At ``cfl = 0.2`` the time error must not masquerade as the spatial one."""
    row = time_split(5, ncells=8, cfl=CFL)
    assert row["diverged"] is False
    assert row["time_share"] is not None
    assert row["time_share"] <= 0.10, row
    # E_space and E_total must bracket the time error (exact semi-discrete split)
    assert row["E_time"] <= row["E_total"]


def test_divergence_is_classified_not_fitted():
    """A blown-up run is flagged, excluded from the share, and never ``inf``."""
    row = time_split(4, ncells=8, cfl=8.0)
    assert row["diverged"] is True
    assert row["time_share"] is None


def test_cfl_budget_reports_window_and_blow_up():
    budget = cfl_budget(2, ncells=8, grid=(0.2, 1.0, 4.0))
    assert budget["cfl_time_share_10"] == 0.2
    assert budget["cfl_divergence"] == 4.0
    shares = [r["time_share"] for r in budget["rows"]]
    assert shares[1] > shares[0]                    # larger cfl, larger share
    assert shares[2] is None                        # divergent row has no share


@pytest.mark.skipif(not (OUT / "metrics.json").exists(),
                    reason="stored artifacts not generated in this checkout")
def test_stored_metrics_and_rendered_doc_agree():
    """The report page is generated from the metrics -- no hand-copied numbers."""
    metrics = json.loads((OUT / "metrics.json").read_text())
    assert metrics["verdict"] in {"PASS", "FAIL", "NOT RUN"}
    assert metrics["barrier"]["table"] == {
        str(s): max_order_for_stages(int(s)) for s in metrics["barrier"]["table"]}
    assert set(metrics["checks"]) >= {"S1_barrier", "S5_cfl_budget"}
    assert max(MAX_ORDER_FOR_STAGES) == 17
    assert all(tab.order <= max_order_for_stages(tab.stages)
               for tab in TABLEAUS.values())
    if DOC.exists():
        text = DOC.read_text()
        for r in metrics["rungs"]:
            if r["fitted_order"] is not None and r["degree"] in (5, 6):
                assert f"{r['fitted_order']:.3f}" in text, r["degree"]