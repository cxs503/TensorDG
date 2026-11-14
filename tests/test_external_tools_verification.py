"""Tests for the KPI-TOOLS-5 external-tool probes (TensorDG side).

The probes assert *tool usability on real DG surfaces*; the tests here assert the
report contract (schema, verdicts, no silent skips) and re-check the two strongest
individual claims (autodiff energy identity, symbolic volume matrix) independently.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordg.external_tools_verification import (
    PROBES,
    REQUIRED_CLASSES,
    probe_autodiff,
    probe_spectral_analysis,
    render_markdown,
    run_all,
)


@pytest.fixture(scope="module")
def report():
    return run_all()


def test_report_schema_and_pass(report):
    assert report["anchor"] == "KPI-TOOLS-5"
    assert report["required_classes"] == REQUIRED_CLASSES
    assert report["tool_classes"] >= REQUIRED_CLASSES
    assert report["real_classes"] == report["tool_classes"]
    assert report["verdict"] == "PASS"
    keys = {"tool_class", "module", "version", "metric", "observed", "reference",
            "abs_err", "limit", "verdict", "detail"}
    for record in report["probes"]:
        assert keys <= set(record)
        assert record["verdict"] == "REAL"
        assert record["abs_err"] <= record["limit"]
        assert record["metric"] and record["detail"]


def test_every_documented_tool_class_is_probed(report):
    assert {record["tool_class"] for record in report["probes"]} == set(PROBES)
    assert len({record["module"] for record in report["probes"]}) == len(report["probes"])


def test_missing_tool_is_reported_not_skipped():
    report = run_all(disabled={"symbolic_math"})
    records = {record["tool_class"]: record for record in report["probes"]}
    assert records["symbolic_math"]["verdict"] == "MISSING"
    assert records["symbolic_math"]["module"] == "sympy"
    assert math.isnan(records["symbolic_math"]["abs_err"])
    assert report["real_classes"] == len(PROBES) - 1
    assert report["verdict"] == "FAIL"


def test_raising_probe_is_reported_as_missing(monkeypatch):
    def boom(_module):
        raise RuntimeError("tool unusable")

    monkeypatch.setitem(PROBES, "visualization", boom)
    report = run_all()
    records = {record["tool_class"]: record for record in report["probes"]}
    assert records["visualization"]["verdict"] == "MISSING"
    assert "RuntimeError" in records["visualization"]["detail"]
    assert report["verdict"] == "FAIL"


def test_autodiff_energy_identity_is_exact():
    import torch as _torch

    record = probe_autodiff(_torch)
    assert record["verdict"] == "REAL"
    assert record["abs_err"] <= 1e-12
    # the dissipation rate must be negative (SIPG stiffness is SPD)
    assert record["observed"] < 0.0
    assert record["observed"] == pytest.approx(record["reference"], rel=1e-12)


def test_symbolic_volume_matrix_matches_exact_rationals():
    import sympy as sp

    from tensordg.advection_1d import _operators

    _, _, _, volume_float, _, _ = _operators(3, torch.device("cpu"), torch.float64)
    xi = sp.symbols("xi", real=True)
    basis = [sp.legendre(k, xi) for k in range(4)]
    expected = sp.Matrix(4, 4, lambda i, j: sp.integrate(
        sp.expand(basis[j] * sp.diff(basis[i], xi)), (xi, -1, 1)))
    actual = volume_float.numpy()
    for i in range(4):
        for j in range(4):
            assert float(expected[i, j]) == pytest.approx(float(actual[i, j]), abs=1e-13)


def test_spectral_probe_measures_dispersion_only():
    record = probe_spectral_analysis(__import__("numpy").fft)
    assert record["verdict"] == "REAL"
    assert record["observed"] == pytest.approx(1.0, abs=1e-6)
    assert "leakage" in record["detail"]


def test_markdown_numbers_come_from_the_report(report):
    markdown = render_markdown(report)
    assert f"real_classes={report['real_classes']}" in markdown
    for record in report["probes"]:
        assert record["tool_class"] in markdown
        if record["abs_err"] != 0.0:
            assert f"{record['abs_err']:.6e}" in markdown
    assert "| MISSING |" not in markdown  # no tool row degraded in this run


def test_verdict_threshold_is_enforced(monkeypatch):
    real_probe = PROBES["spectral_analysis"]

    def rigged(module):
        record = real_probe(module)
        record["abs_err"] = record["limit"] * 10.0
        record["verdict"] = "REAL"  # a lying probe must not survive run_all
        return record

    monkeypatch.setitem(PROBES, "spectral_analysis", rigged)
    report = run_all()
    records = {record["tool_class"]: record for record in report["probes"]}
    # run_all re-derives the verdict from abs_err vs limit, so a self-labelled
    # "REAL" record with an out-of-tolerance error is downgraded to FAIL
    assert records["spectral_analysis"]["abs_err"] > records["spectral_analysis"]["limit"]
    assert records["spectral_analysis"]["verdict"] == "FAIL"
    assert report["verdict"] == "FAIL"