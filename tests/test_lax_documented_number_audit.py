"""Gates for the documented-Lax-number audit (no solver re-run: artifact only).

The audit driver (``benchmarks/lax_documented_number_audit.py``) regenerates
``docs/benchmarks/results/lax_documented_number_audit/metrics.json`` + the
provenance cache in ~3 min.  These tests bind every claim in
``docs/CASE-LAX-DG-documented-number-audit.md`` to that committed artifact, so a
later solver change cannot silently make the document wrong again.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "benchmarks" / "results" / "lax_documented_number_audit"
SPEC = importlib.util.spec_from_file_location(
    "lax_documented_number_audit", ROOT / "benchmarks" / "lax_documented_number_audit.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def _metrics() -> dict:
    return json.loads((OUT / "metrics.json").read_text())


def test_artifact_matches_the_registered_configuration():
    m = _metrics()
    assert (m["config"]["ncells"], m["config"]["degree"], m["config"]["cfl"]) == (400, 3, 0.35)
    assert m["config"]["final_time"] == 0.13
    assert "".join(["p", str(m["config"]["degree"])]) == "p3"


def test_tolerance_is_never_copied_by_hand():
    from tensordg.cases.lax_1d import lax_parameters

    assert _metrics()["tolerance"] == lax_parameters()["acceptance_threshold"] == 0.03


def test_documented_number_is_reproduced_by_the_publisher_commit():
    """Provenance: 2.769420 % must be exactly the q=8 reading of the old solver."""
    replay = json.loads((OUT / "historical_replay.json").read_text())
    assert replay["commit"].startswith("432bef0")
    assert abs(replay["readings"]["8"]["density"] - 0.027694203386353103) < 1e-12
    assert abs(replay["readings"]["8"]["velocity"] - 0.016052703865463716) < 1e-12
    assert abs(replay["readings"]["8"]["pressure"] - 0.009193692240376066) < 1e-12
    m = _metrics()
    assert abs(m["documented_number"] - 0.0276942) < 1e-12


def test_documented_number_is_not_reproducible_by_main():
    """The published number must not be presented as a current-code reading."""
    f = _metrics()["findings"]
    assert f["documented_reproducible_from_main"] is False
    assert f["nearest_current_reading_gap"] > 1e-3          # > 0.1 percentage point


def test_default_documented_metric_does_not_match_the_published_number():
    m = _metrics()
    default = m["metric_sweep"]["8"]["density"]
    assert abs(default - m["documented_number"]) > 1e-3
    assert abs(default - m["findings"]["default_rule_value"]) < 1e-15


def test_fixed_gauss_rule_verdict_is_quadrature_dependent():
    """The 3 % verdict must not be decidable by the fixed rule alone."""
    m = _metrics()
    low, high = m["findings"]["quadrature_spread_density"]
    assert high - low > 3e-3                                # > 0.3 percentage point
    verdicts = {max(row.values()) < m["tolerance"] for row in m["metric_sweep"].values()}
    assert verdicts == {True, False}                        # both PASS and FAIL appear


def test_quadrature_free_reading_is_an_honest_fail():
    m = _metrics()
    f = m["findings"]
    assert f["converged_worst_field"] == "density"
    assert f["converged_value"] > m["tolerance"]
    assert f["converged_pass"] is False
    # 0.8 % above the bar: small but unambiguous margin
    assert 0.0300 < f["converged_value"] < 0.0308


def test_converged_evaluator_is_self_converged():
    m = _metrics()
    conv = m["converged"]
    fine, coarse = conv["640"]["density"], conv["160"]["density"]
    assert abs(fine - coarse) < 1e-5                       # < 0.001 percentage point
    detected = conv["detected_breaks_640"]["density"]
    assert abs(detected - fine) < 1e-5                     # insensitive to breakpoint source


def test_independent_implementation_agrees_with_the_committed_quadrature_audit():
    """Two independent wave-aligned evaluators must land on the same number."""
    m = _metrics()
    cross = m["cross_check_quadrature_audit"]
    assert cross is not None, "committed lax_a0_quadrature_audit artifact is required"
    assert abs(cross - m["converged"]["640"]["density"]) < 1e-5


def test_public_document_claims_follow_the_artifact():
    m = _metrics()
    readme = (ROOT / "README.md").read_text()
    tutorial = (ROOT / "docs" / "tutorials" / "nonlinear-dg-benchmarks.md").read_text()
    expand = (ROOT / "expand_audit_tensordg.md").read_text()
    for text in (readme, tutorial, expand):
        assert "2.769420" not in text, "stale pre-audit Lax number is still published"
        assert "CASE-LAX-DG-documented-number-audit" in text
    assert f"{100 * m['converged']['640']['density']:.3f}" in readme


def test_document_is_rendered_from_the_artifact():
    metrics = _metrics()
    rendered = AUDIT.render(metrics)
    committed = (ROOT / "docs" / "CASE-LAX-DG-documented-number-audit.md").read_text()
    assert rendered.strip() == committed.strip()