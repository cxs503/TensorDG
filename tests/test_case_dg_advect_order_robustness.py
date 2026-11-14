"""Invariant tests for the CASE-DG-ADVECT-ORDER round-4 robustness audit.

These tests do not merely assert that a file exists: every reported figure is
recomputed from the *raw* ladders stored in ``metrics.json`` (our runs) and from
``validation/advection_order/ref_table.json`` (the mechanically parsed paper
table), so a silent edit of a verdict or a loosened tolerance fails the suite.

    PYTHONPATH=. pytest -q tests/test_case_dg_advect_order_robustness.py
"""
from __future__ import annotations

import json
import math
import pathlib

import pytest

BASE = pathlib.Path("validation/advection_order_robustness")
METRICS = BASE / "metrics.json"
SUMMARY = BASE / "summary.md"
LADDERS = BASE / "ladders.csv"
CFL_CSV = BASE / "cfl_window.csv"
SNAPSHOT = BASE / "solution_snapshot.json"
DOC = pathlib.Path("docs/CASE-DG-ADVECT-ORDER-robustness.md")
FIG = pathlib.Path("docs/figures/advorder_robust_estimators.png")
FIG_SOL = pathlib.Path("docs/figures/advorder_robust_solution.png")
REF = pathlib.Path("validation/advection_order/ref_table.json")


@pytest.fixture(scope="module")
def m():
    if not METRICS.exists():
        pytest.skip("round-4 metrics not assembled yet")
    return json.loads(METRICS.read_text())


@pytest.fixture(scope="module")
def ref():
    return json.loads(REF.read_text())


def _fit_order(cells, l2):
    import numpy as np
    x = np.log(np.asarray(cells, float))
    y = np.log(np.asarray(l2, float))
    return float(-np.polyfit(x, y, 1)[0])


def test_artefacts_exist():
    for p in (METRICS, SUMMARY, LADDERS, CFL_CSV, DOC, FIG, FIG_SOL):
        assert p.exists(), f"missing artefact {p}"
    assert FIG.stat().st_size > 1000 and FIG_SOL.stat().st_size > 1000


def test_thresholds_not_loosened(m):
    """The predeclared tolerance must be the documented 0.10 order units."""
    assert m["thresholds"]["RATE_TOL"] == pytest.approx(0.10)
    assert m["thresholds"]["RATIO_SPREAD_TOL"] == pytest.approx(0.05)


def test_verdict_matches_checks(m):
    assert m["verdict"] in ("PASS", "FAIL")
    assert m["verdict"] == ("PASS" if all(m["checks"].values()) else "FAIL")
    assert m["failed"] == [k for k, v in m["checks"].items() if not v]
    strict = all(m["checks"].values()) and all(m["aux_checks"].values())
    assert m["verdict_strict_all_checks"] == ("PASS" if strict else "FAIL")
    assert m["aux_failed"] == [k for k, v in m["aux_checks"].items() if not v]


def test_roundoff_screen_is_declared_and_consistent(m):
    """The screen must be predeclared, positive, and monotonically increasing."""
    for key, rec in m["base"].items():
        floors = [(r["ncells"], r["roundoff_floor"], r["above_floor"])
                  for r in rec["rows"]]
        vals = [f for _, f, _ in floors]
        assert vals == sorted(vals), f"{key}: floor not monotone in N"
        eps = 2.220446049250313e-16
        for n, f, ok in floors:
            assert f > eps
            assert ok is bool(rec["rows"][[r["ncells"] for r in rec["rows"]].index(n)]["l2"] > f)


def test_reference_provenance_present(m):
    prov = m["reference"]["provenance"]
    assert len(prov["table_block_sha256"]) == 64
    assert prov["eprint_url"].startswith("https://")
    assert "10.1090/mcom/4037" in m["reference"]["citation"]["doi"]


def test_b1_estimators_recomputed_from_ladders(m):
    """Each estimator and every deviation is reproduced from the stored ladder."""
    tol = m["thresholds"]["RATE_TOL"]
    for key, rec in m["B1_estimator_independence"]["detail"].items():
        rows = m["base"][key]["rows"]
        keep = [r for r in rows if r["above_floor"]] or rows
        assert [r["ncells"] for r in keep] == rec["screened_cells"]
        assert [r["ncells"] for r in rows if not r["above_floor"]] == \
            rec["roundoff_screened_out"]
        cells = [r["ncells"] for r in keep]
        l2 = [r["l2"] for r in keep]
        assert rec["fit_all_raw"] == pytest.approx(
            _fit_order([r["ncells"] for r in rows], [r["l2"] for r in rows]),
            abs=1e-9)
        fit_all = _fit_order(cells, l2)
        fit3 = _fit_order(cells[-3:], l2[-3:])
        pair = math.log(l2[-2] / l2[-1]) / math.log(cells[-1] / cells[-2])
        assert rec["fit_all"] == pytest.approx(fit_all, abs=1e-9)
        assert rec["fit_finest3"] == pytest.approx(fit3, abs=1e-9)
        assert rec["finest_pair"] == pytest.approx(pair, abs=1e-9)
        assert rec["dev_fit_all"] == pytest.approx(abs(fit_all - rec["ref_fit_all"]),
                                                   abs=1e-9)
        spread = max(fit_all, fit3, pair) - min(fit_all, fit3, pair)
        assert rec["estimator_spread"] == pytest.approx(spread, abs=1e-9)
        assert rec["pass"] is bool(spread <= tol and rec["dev_fit_all"] <= tol
                                   and abs(fit3 - rec["ref_fit_all"]) <= tol
                                   and abs(pair - rec["ref_fit_all"]) <= tol)


def test_b1_ref_fit_reproduced_from_paper_table(m, ref):
    """The reference estimator is a fit of the paper's own published column."""
    cols = {c["degree"]: c["name"] for c in ref["columns"]}
    cells = [row["N"] for row in ref["rows"]]
    for d, name in cols.items():
        l2r = [row[name]["l2"] for row in ref["rows"]]
        rec = m["reference"]["estimators"][f"P{d}"]
        assert rec["column"] == name
        assert rec["fit_all"] == pytest.approx(_fit_order(cells, l2r), abs=1e-9)
        assert rec["printed"] == [row[name]["order_printed"] for row in ref["rows"]]


def test_b2_rows_cover_every_printed_order(m, ref):
    """B2 must judge each (degree, mesh) where the paper prints an order."""
    printed = {(c["degree"], row["N"])
               for c in ref["columns"] for row in ref["rows"]
               if row[c["name"]]["order_printed"] is not None}
    judged = {(r["degree"], r["N"]) for r in m["B2_per_mesh_vs_paper"]["rows"]}
    assert judged == printed, "B2 does not cover the paper's printed orders"
    for r in m["B2_per_mesh_vs_paper"]["rows"]:
        gap = abs(r["order_ours"] - r["order_paper"])
        assert r["gap"] == pytest.approx(gap, abs=1e-9)
        assert r["pass"] is bool(gap <= m["thresholds"]["RATE_TOL"])


def test_b2_per_mesh_orders_recomputed(m):
    for r in m["B2_per_mesh_vs_paper"]["rows"]:
        rows = m["base"][f"P{r['degree']}"]["rows"]
        cells = [x["ncells"] for x in rows]
        l2 = [x["l2"] for x in rows]
        k = cells.index(r["N"])
        ours = math.log(l2[k - 1] / l2[k]) / math.log(cells[k] / cells[k - 1])
        assert r["order_ours"] == pytest.approx(ours, abs=1e-9)


def test_b3_gates_the_declared_window(m):
    sweep = m["B3_cfl_window"]["sweep"]
    gated = m["setup"]["cfl_gated"]
    judged = sorted({float(k.split("=")[1].split("/")[0])
                     for k in m["B3_cfl_window"]["detail"]})
    assert judged == sorted(gated)
    for key, rec in m["B3_cfl_window"]["detail"].items():
        gap = abs(rec["order"] - rec["ref_order"]) if math.isfinite(rec["order"]) else None
        if gap is not None:
            assert rec["gap"] == pytest.approx(gap, abs=1e-9)
        assert rec["pass"] is bool(rec["finite"] and gap is not None
                                   and gap <= m["thresholds"]["RATE_TOL"])
    # the declared validity edge is derivable from the sweep, not asserted
    tol = m["thresholds"]["RATE_TOL"]
    ok_cfls = [float(c) for c, rec in sweep.items()
               if all(r["all_finite"] and r["gap"] <= tol for r in rec.values())]
    assert m["B3_cfl_window"]["validated_edge_cfl"] == (
        max(ok_cfls) if ok_cfls else None)


def test_b3_order_is_cfl_insensitive_inside_window(m):
    """The rate is a spatial property: order across the gated CFL window must agree."""
    sweep = m["B3_cfl_window"]["sweep"]
    for d in m["setup"]["degrees"]:
        lo = f"{sorted(m['setup']['cfl_gated'])[0]:g}"
        a = sweep[lo][f"P{d}"]["order"]
        b = sweep["0.2"][f"P{d}"]["order"]
        assert abs(a - b) <= m["thresholds"]["RATE_TOL"], (d, a, b)


def test_b4_ratios_recomputed(m, ref):
    cols = {c["degree"]: c["name"] for c in ref["columns"]}
    for key, rec in m["B4_prefactor_constancy"]["detail"].items():
        d = int(key[1:])
        ours = [r["l2"] for r in m["base"][key]["rows"] if r["above_floor"]]
        paper = [row[cols[d]]["l2"] for row in ref["rows"]
                 if row["N"] in rec["meshes"]]
        assert len(ours) == len(paper) == len(rec["meshes"])
        ratio = [a / b for a, b in zip(ours, paper)]
        assert rec["ratios"] == pytest.approx(ratio, rel=1e-9)
        assert rec["mean_ratio"] == pytest.approx(sum(ratio) / len(ratio), rel=1e-9)
        spread = (max(ratio) - min(ratio)) / (sum(ratio) / len(ratio))
        assert rec["relative_spread"] == pytest.approx(spread, rel=1e-9)


def test_no_nan_in_reported_ladders(m):
    for key, rec in m["base"].items():
        for row in rec["rows"]:
            assert row["finite"] is True, f"{key} N={row['ncells']} not finite"
            assert row["l2"] > 0.0
            assert row["steps"] > 0


def test_summary_and_doc_are_identical(m):
    assert SUMMARY.read_text() == DOC.read_text(), (
        "docs/ report and validation summary diverged")
    txt = SUMMARY.read_text()
    assert "是否 <3%" in txt or "<3%" in txt
    for needle in ("B1", "B2", "B3", "B4"):
        assert needle in txt


def test_solution_snapshot_order_matches_report():
    if not SNAPSHOT.exists():
        pytest.skip("solution snapshot not generated")
    s = json.loads(SNAPSHOT.read_text())
    errs = s["errors"]
    # higher degree must be more accurate on the same mesh
    vals = [errs[f"P{d}"]["l2"] for d in (1, 2, 3, 4)]
    assert vals == sorted(vals, reverse=True), vals