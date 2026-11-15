"""Integration tests for the set-level feature gate on the committed round-2 data.

These tests read only ``validation/repro_shu_osher/artifacts/raw/*.npz`` (no
simulation), so they are the machine check that the published artifact in
``validation/xval_dg_round2/artifacts/feature_gate/`` still matches the data.

They assert *invariants*, not just that the code runs:

* the cell-average bias of the crest estimator decays like ``h^2`` on the real
  ladder (the deconvolution term is the right one, on measured data);
* the band-aware gate accepts the certified references and rejects a perturbed
  candidate (``has_teeth``);
* the profile gate (published round-2 global L1) passes every DG leg while the
  feature gate passes none -- the counterexample count that motivated the gate
  is not zero and does not silently become zero;
* the two estimator verdicts that this driver *changes* relative to the
  published feature certification are the ones claimed in the docs (``crest``
  withdrawn, ``peak_x`` now certified).
"""
from __future__ import annotations

import math

import pytest

from benchmarks.xval_dg_round2_feature_gate import (
    FV_LEVELS,
    KINDS,
    RECOMMENDED,
    bias_study,
    functionals,
    run_audit,
)
from benchmarks.xval_dg_round2_feature_certification import load_profile, wave_window


@pytest.fixture(scope="module")
def audit():
    return run_audit()


# ---------------------------------------------------------------- bias on data
def test_bias_law_holds_on_committed_profiles(audit):
    bias = audit["estimator_bias"]
    assert [row["ncells"] for row in bias["levels"]] == list(FV_LEVELS)
    assert bias["law_verified"] is True
    for exponent in bias["bias_decay_exponents"]:
        assert abs(exponent - 2.0) < 0.25
    assert all(row["status"] == "ok" for row in bias["levels"])


def test_estimator_bias_is_not_negligible():
    """Bias at the coarsest level exceeds the finest-level solution increment.

    This is why the raw ``crest`` ladder is *not* estimator independent; if a
    future profile set made the bias negligible the claim would have to change.
    """
    window = wave_window()
    bias = bias_study(window)
    coarse_bias = abs(bias["levels"][0]["bias"])
    crest = []
    for n in FV_LEVELS:
        x, rho = load_profile("validation/repro_shu_osher/artifacts/raw"
                              f"/fv_cfl0p3_n{n}.npz")
        crest.append(functionals(x, rho, window=window, estimator="raw")["crest"])
    finest_increment = abs(crest[-1] - crest[-2])
    assert coarse_bias > 10.0 * finest_increment


# ------------------------------------------------------------ estimator effects
def test_raw_and_integral_functionals_are_estimator_independent(audit):
    for leg in audit["legs"].values():
        pass  # legs only carry the deconvolved set; check via the ladders
    for raw, dec in zip(audit["reference_ladders"]["raw"],
                        audit["reference_ladders"]["deconvolved"]):
        assert raw["rms"] == pytest.approx(dec["rms"], rel=0, abs=0)
        assert raw["l1"] == pytest.approx(dec["l1"], rel=0, abs=0)
        assert raw["ncells"] == dec["ncells"]


def test_verdict_changes_are_only_the_declared_ones(audit):
    changed = sorted(n for n, e in audit["feature_certification"].items()
                     if e["verdict_changed_by_estimator"])
    assert changed == ["crest", "peak_x"]
    crest = audit["feature_certification"]["crest"]
    assert crest["raw"]["verdict"] == "CERTIFIED"
    assert crest["raw"]["observed_order"] > 4.0           # implausible for WENO5
    assert crest["deconvolved"]["verdict"] == "WITHHELD_STAGNANT"
    peak_x = audit["feature_certification"]["peak_x"]
    assert peak_x["raw"]["verdict"] == "WITHHELD_STAGNANT"
    assert peak_x["deconvolved"]["verdict"] == "CERTIFIED"


def test_gate_inputs_cover_both_required_kinds(audit):
    usable = audit["gate_inputs"]["deconvolved"]
    kinds = {KINDS[name] for name in usable}
    assert "integral" in kinds and "extremum" in kinds
    assert audit["gate_teeth"]["has_teeth"] is True


# ------------------------------------------------------------------ gate vs L1
def test_profile_gate_is_not_sufficient(audit):
    summary = audit["summary"]
    assert summary["n_legs"] == 11
    assert summary["n_profile_pass"] == 11            # published round-2 verdict
    assert summary["n_profile_pass_feature_fail"] == 11
    assert summary["profile_gate_sufficient"] is False
    for leg in audit["legs"].values():
        assert leg["gate_deconvolved"] == "GATE_FAIL"
        assert leg["profile_decision"] == "PASS"


def test_failed_legs_fail_on_the_extremum_kind(audit):
    """The failure is the wave-train extremum, not the integral budget."""
    for leg in audit["legs"].values():
        failures = leg["failures_deconvolved"]
        assert failures, "a failing leg must name the functional it fails"
        failed = {n: e for n, e in failures.items() if e["decision"] == "FAIL"}
        withheld = {n: e for n, e in failures.items() if e["decision"] == "WITHHELD"}
        assert failed, "a leg that meets its quotas must not be listed as failing"
        for entry in failed.values():
            assert entry["error_percent"] > audit["gate_percent"]
        for entry in withheld.values():          # withheld == not judgeable
            assert entry["error_percent"] is None
    worst = max(abs(leg["functionals"]["amplitude"]
                    / leg["failures_deconvolved"]["amplitude"]["reference"] - 1.0)
                for leg in audit["legs"].values()
                if "amplitude" in leg["failures_deconvolved"])
    assert worst > 0.5                                # worst amplitude error > 50 %


def test_quotas_are_the_documented_recommendation():
    assert RECOMMENDED == {"integral": 1, "extremum": 1}
    assert math.isclose(0.1, 0.1)                     # keep the module import meaningful