"""L3 integration gates for the certified-reference engine (XVAL-DG-ROUND2).

Three things the maturity gate could not see in the previous delivery are pinned
here as machine checks:

1. **Reachability** -- the engine's public names (and the four ``convergence``
   names that were missing before it) resolve from the package root, and the
   contract gate's own ratio is 1.0.  This is what made ``from tensordg import
   certify`` an ``ImportError`` before.
2. **Case-API integration** -- ``tensordg.cases.shock_entropy.gate_report``
   accepts a certification and reports the ``judge`` decision (PASS /
   PASS_MARGINAL / FAIL / WITHHELD) instead of a bare ``err < tau``, on a profile
   it solved itself.  The leg here is tiny (L1 N=40) so the test is a wiring check;
   the numeric benchmark lives in ``benchmarks/xval_dg_round2_certified.py``.
3. **Gate-validity refusal** -- a reference whose band consumes the gate must make
   the case API report ``WITHHELD`` rather than a pass, i.e. the integration cannot
   be bypassed to manufacture a green verdict.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "validation/repro_shu_osher/artifacts/raw"


# --------------------------------------------------------------- 1. reachability
def test_root_export_reaches_the_engine_and_the_contract_gate_is_green():
    import tensordg
    from tensordg import (certify, judge, classify, audit_ladder,  # noqa: F401
                          band_coverage_study, digest_levels, rel_l1, gci,
                          certified_reference_coarsen,
                          certified_reference_observed_order,
                          certified_reference_richardson_extrapolate,
                          semidiscrete_operator, exact_time_solution)

    assert certify.__module__ == "tensordg.certified_reference"
    assert judge.__module__ == "tensordg.certified_reference"
    # the aliases must be the *same objects*, not look-alikes
    assert certified_reference_observed_order.__name__ == "observed_order"
    assert "certify" in tensordg.__all__ and "judge" in tensordg.__all__
    assert len(tensordg.__all__) == len(set(tensordg.__all__))

    spec = importlib.util.spec_from_file_location(
        "api_contract", ROOT / "tools/api_contract.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    contract = tool.audit()
    assert contract["coverage"]["ratio"] == 1.0
    assert contract["coverage"]["modules_with_zero_exports"] == []
    assert contract["identity"]["violations"] == []


# ------------------------------------------------------- 2./3. case-API wiring
def _synthetic_certification(reference_size: int = 40, tolerance: float = 5.0) -> dict:
    """A certification with a *known* band, built from a synthetic power-law ladder."""
    from tensordg.certified_reference import certify

    levels = []
    for n in (reference_size // 4, reference_size // 2, reference_size):
        x = (np.arange(n) + 0.5) / n
        levels.append({"ncells": n, "field": 1.0 + 0.3 * x + 0.05 * n ** -1.0})
    cert = certify(levels, tolerance_percent=tolerance)
    assert cert["verdict"] == "CERTIFIED", cert["verdict"]
    return cert


def test_case_api_reports_the_judge_decision_on_a_profile_it_solved():
    from tensordg.cases import shock_entropy
    from tensordg.certified_reference import judge

    cert = _synthetic_certification(40)
    report = shock_entropy.gate_report(40, degree=1, cfl=0.2, certification=cert,
                                       tolerance_percent=5.0, return_profile=True)
    verdict = report["certified_judgement"]
    assert verdict["decision"] in ("PASS", "PASS_MARGINAL", "FAIL", "WITHHELD")
    # the case API must not invent its own decision: it is judge()'s, on the same field
    assert verdict["rel_l1_percent"] == pytest.approx(
        judge(report["rho"], cert, tolerance_percent=5.0)["rel_l1_percent"], rel=1e-12)
    assert report["passes_gate"] is (verdict["decision"] == "PASS")
    assert report["reference"]["band_percent"] == pytest.approx(cert["band_percent"])
    # the profile handed back is the one that was judged (no second solve, no swap)
    assert report["rho"].shape == (40,)
    assert abs(float(np.mean(report["rho"][report["x"] <= -3.0])) - 3.857143) < 0.01


def test_case_api_refuses_a_reference_whose_band_consumes_the_gate():
    from tensordg.cases import shock_entropy

    cert = _synthetic_certification(40)
    # ask the same certificate to decide a gate far tighter than its own band
    tight = 1e-4
    report = shock_entropy.gate_report(40, degree=1, cfl=0.2, certification=cert,
                                       tolerance_percent=tight)
    assert report["certified_judgement"]["decision"] == "WITHHELD"
    assert report["passes_gate"] is False


def test_gate_report_rejects_two_competing_references():
    from tensordg.cases import shock_entropy

    cert = _synthetic_certification(40)
    with pytest.raises(ValueError, match="not both"):
        shock_entropy.gate_report(40, degree=1, cfl=0.2, certification=cert,
                                  reference=(np.linspace(-5, 5, 40), np.ones(40)))


# --------------------------------------------------------- real-ladder contract
@pytest.mark.skipif(not RAW.exists(), reason="committed FV ladder not in this checkout")
def test_committed_ladder_certifies_and_binds_to_input_bytes():
    from tensordg.certified_reference import certify

    levels = []
    for n in (800, 1600, 3200):
        with np.load(RAW / f"fv_cfl0p3_n{n}.npz", allow_pickle=True) as d:
            levels.append({"ncells": n, "field": np.asarray(d["rho"], dtype=float)})
    cert = certify(levels, tolerance_percent=5.0)
    assert cert["verdict"] == "CERTIFIED"
    assert 1.0 < cert["observed_order_used"] < 2.0
    assert cert["band_share_of_gate"] <= 1.0 / 3.0
    # the certificate is byte-bound: perturbing one level changes its digest
    import copy

    from tensordg.certified_reference import digest_levels
    h0 = digest_levels(levels)
    broken = copy.deepcopy(levels)
    broken[-1]["field"] = broken[-1]["field"] + 1e-9
    assert digest_levels(broken) != h0
    assert json.dumps(cert["input_sha256"]) == json.dumps(h0)