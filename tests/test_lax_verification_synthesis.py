"""CASE-LAX-DG · verification synthesis contract tests.

These tests assert that the synthesis report (``docs/CASE-LAX-DG-verification-synthesis.md``)
is a faithful, machine-checkable summary of all committed artifacts.  They do **not**
re-run any solver; they read the already-committed ``metrics.json`` files and verify
that every number in the synthesis matches the source of truth.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SYNTHESIS = ROOT / "docs" / "CASE-LAX-DG-verification-synthesis.md"
POOL_METRICS = ROOT / "docs" / "benchmarks" / "results" / "lax_case_pool" / "metrics.json"
CLOSURE_METRICS = ROOT / "docs" / "benchmarks" / "results" / "lax_resolution_closure" / "metrics.json"
CERTIFIED_METRICS = ROOT / "docs" / "benchmarks" / "results" / "lax_certified_configuration" / "metrics.json"


@pytest.fixture(scope="module")
def synthesis_text():
    if not SYNTHESIS.exists():
        pytest.skip(f"synthesis not yet committed: {SYNTHESIS.relative_to(ROOT)}")
    return SYNTHESIS.read_text()


@pytest.fixture(scope="module")
def pool():
    if not POOL_METRICS.exists():
        pytest.skip("lax_case_pool artifact missing")
    return json.loads(POOL_METRICS.read_text())


@pytest.fixture(scope="module")
def closure():
    if not CLOSURE_METRICS.exists():
        pytest.skip("lax_resolution_closure artifact missing")
    return json.loads(CLOSURE_METRICS.read_text())


@pytest.fixture(scope="module")
def certified():
    if not CERTIFIED_METRICS.exists():
        pytest.skip("lax_certified_configuration artifact missing")
    return json.loads(CERTIFIED_METRICS.read_text())


# --------------------------------------------------------------------------- #
# 1. synthesis file exists and has required sections
# --------------------------------------------------------------------------- #
def test_synthesis_file_exists():
    assert SYNTHESIS.exists(), f"synthesis not found at {SYNTHESIS.relative_to(ROOT)}"


def test_synthesis_has_required_sections(synthesis_text):
    for section in (
        "① 案例介绍", "② 方法", "③ 验证证据总览", "④ 计算结果",
        "⑤ 判据映射", "⑥ 结果图", "⑦ 注意点", "⑧ 误差判定", "⑨ 闭合建议",
    ):
        assert section in synthesis_text, f"missing section: {section}"


# --------------------------------------------------------------------------- #
# 2. key numbers match committed artifacts
# --------------------------------------------------------------------------- #
def test_a0_production_leg_matches_pool(pool):
    acceptance = pool["acceptance"]
    assert acceptance["production_leg"] == "p3.N400"
    leg = acceptance["per_leg"]["p3.N400"]
    assert leg["l2"]["density"] == pytest.approx(0.0311556, rel=1e-3)


def test_resolution_closure_n800_matches(closure):
    n800 = next(leg for leg in closure["ladder"] if leg["cells"] == 800)
    assert n800["l2"]["density"] == pytest.approx(0.02315, rel=1e-2)
    assert n800["l2"]["density"] < 0.03


def test_certified_config_is_p2_n800(certified):
    cert = certified["certification"]["certified"]
    assert cert["leg"] == "p2.N800"
    assert cert["cells"] == 800
    assert cert["degree"] == 2
    assert cert["dof"] == 2400
    assert cert["worst_l2"] == pytest.approx(0.02570, rel=1e-2)
    assert cert["worst_l2"] < 0.03


def test_certified_worst_field_is_density(certified):
    cert = certified["certification"]["certified"]
    leg = next(leg for leg in certified["legs"] if leg["leg"] == cert["leg"])
    worst = max(leg["l2_errors"].values())
    assert leg["l2_errors"]["density"] == pytest.approx(worst)


# --------------------------------------------------------------------------- #
# 3. verdict mapping is consistent
# --------------------------------------------------------------------------- #
def test_production_leg_status_is_fail(pool):
    leg = pool["acceptance"]["per_leg"]["p3.N400"]
    assert leg["status"] == "FAIL"


def test_ladder_l1_errors_decrease_with_resolution(pool):
    """L1 errors should decrease as resolution increases (monotone convergence)."""
    legs = pool["acceptance"]["per_leg"]
    # p1 series: N=100, 200, 400
    p1_density_l1 = [legs[f"p1.N{n}"]["l1"]["density"] for n in [100, 200, 400]]
    assert p1_density_l1[0] > p1_density_l1[1] > p1_density_l1[2]


def test_closure_gate_is_pass(closure):
    assert closure["gate"]["status"] == "PASS"


def test_certified_verdict_is_pass(certified):
    assert certified["certification"]["status"] == "PASS"


# --------------------------------------------------------------------------- #
# 4. synthesis mentions key facts
# --------------------------------------------------------------------------- #
def test_synthesis_mentions_honest_fail(synthesis_text):
    assert "FAIL" in synthesis_text


def test_synthesis_mentions_certified_config(synthesis_text):
    assert "p2.N800" in synthesis_text
    assert "2400" in synthesis_text


def test_synthesis_mentions_closure_n800(synthesis_text):
    assert "N800" in synthesis_text
    assert "2.315" in synthesis_text


def test_synthesis_mentions_production_fail(synthesis_text):
    assert "3.116" in synthesis_text


# --------------------------------------------------------------------------- #
# 5. cross-artifact consistency
# --------------------------------------------------------------------------- #
def test_pool_p3_n400_matches_closure_n400(pool, closure):
    pool_p3n400 = pool["acceptance"]["per_leg"]["p3.N400"]
    closure_n400 = next(leg for leg in closure["ladder"] if leg["cells"] == 400)
    assert pool_p3n400["l2"]["density"] == pytest.approx(
        closure_n400["l2"]["density"], rel=1e-6
    )


def test_all_artifacts_reference_same_tolerance(pool, closure, certified):
    pool_tol = pool["acceptance"]["tolerance"]
    closure_tol = closure["config"].get("tolerance", 0.03)
    certified_tol = certified["certification"]["tolerance"]
    assert pool_tol == pytest.approx(0.03, rel=1e-6)
    assert closure_tol == pytest.approx(0.03, rel=1e-6)
    assert certified_tol == pytest.approx(0.03, rel=1e-6)
