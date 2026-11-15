"""Gates for the CASE-LAX-DG cross-environment band artifact (offline, no solver).

Every assertion re-derives a number from committed evidence: the raw runs
recorded on this host, the legacy probe artifact read out of git, the committed
probe/pool artifacts and the certification certificate.  Nothing re-runs the DG
solver, so the suite stays a fast unit test.
"""
from __future__ import annotations

import json

import pytest

from benchmarks import lax_crossenv_band as X


@pytest.fixture(scope="module")
def metrics():
    return X._load(X.OUT / "metrics.json")


def test_artifact_is_sealed_and_self_digesting(metrics):
    assert metrics["metrics_sha256"] == X.digest(metrics)
    sealed = (X.OUT / "metrics.sha256").read_text().strip()
    assert sealed == metrics["metrics_sha256"]


def test_thread_sweep_is_bitwise_identical(metrics):
    """H-thread control 1: only the thread count changes, nothing else."""
    inv = metrics["thread_invariance"]
    assert inv["legs"], "no thread sweep recorded"
    settings = max(len(set(leg["threads"])) for leg in inv["legs"])
    assert settings >= 3, "the sweep must cover several thread counts on at least one leg"
    for leg in inv["legs"]:
        assert leg["bitwise_identical"] is True
        assert leg["worst_relative_spread"] == pytest.approx(0.0, abs=1e-15)
        assert leg["verdict"] == "PASS"


def test_thread_count_cannot_explain_the_drift(metrics):
    """The drift is >=1e5x larger than anything the thread sweep produces."""
    band = metrics["crossenv_band"]["max_band"]
    spread = metrics["thread_invariance"]["worst_relative_spread"]
    assert spread <= 1e-15
    assert band > spread * 1e5


def test_reproducibility_control_agrees(metrics):
    repro = metrics["reproducibility_control"]
    assert repro["verdict"] == "PASS"
    assert repro["worst_relative_spread"] <= 1e-12
    assert max(repro["thread_settings"]) >= 2 * min(repro["thread_settings"])


def test_legacy_artifact_is_reproduced_bitwise(metrics):
    """H-stale-code falsified: today's code reproduces the legacy values exactly."""
    rows = metrics["crossenv_band"]["legacy_reproduction"]["rows"]
    assert rows
    for row in rows:
        assert row["steps"] == row["legacy_steps"]
        assert row["bitwise_equal"] is True
        assert row["worst_relative_difference"] == pytest.approx(0.0, abs=1e-18)


def test_band_rows_reproduce_from_committed_artifacts(metrics):
    """Every band row must be re-derivable from the two committed artifacts."""
    legacy = X.legacy_probe()
    probe_now = X._load(X.PROBE_OUT / "metrics.json")
    pool = X._load(X.POOL_OUT / "metrics.json")
    fresh = X.committed_pair_rows(legacy, [("regenerated probe artifact (cluster)", probe_now),
                                           ("committed pool artifact", pool)])
    recorded = metrics["crossenv_band"]["rows"]
    assert len(fresh) == len(recorded)
    for new, old in zip(fresh, recorded):
        assert new["leg"] == old["leg"]
        assert new["field"] == old["field"]
        assert new["rel_diff"] == pytest.approx(old["rel_diff"], rel=1e-12)


def test_band_is_the_observed_maximum_and_not_extrapolated(metrics):
    band = metrics["crossenv_band"]
    assert band["max_band"] == pytest.approx(max(r["rel_diff"] for r in band["rows"]))
    assert band["max_band_leg"] == max(band["per_leg"], key=lambda k: band["per_leg"][k])
    # the artifact documents that no step-count law was fitted
    assert "none" in band["model"]
    assert band["band_is_monotone_in_steps"] is False


def test_gate_uses_the_field_max_band(metrics):
    band = metrics["crossenv_band"]
    assert band["max_band"] == pytest.approx(max(band["per_field_max"].values()))
    assert set(band["per_field_max"]) == {"density", "velocity", "pressure"}


def test_verdict_robustness_reproduces_from_the_certificate(metrics):
    cert = X._load(X.CERT_OUT / "metrics.json")
    fresh = X.verdict_robustness(cert, metrics["crossenv_band"]["max_band"])
    recorded = metrics["verdict_robustness"]
    assert [r["leg"] for r in fresh["rows"]] == [r["leg"] for r in recorded["rows"]]
    for new, old in zip(fresh["rows"], recorded["rows"]):
        assert new["worst_l2"] == pytest.approx(old["worst_l2"], rel=1e-12)
        assert new["status"] == old["status"]
        assert new["distance_to_bar"] == pytest.approx(old["distance_to_bar"], rel=1e-12)
        assert new["robust"] == old["robust"]
    assert fresh["tolerance"] == recorded["tolerance"]


def test_tolerance_comes_from_the_case_module(metrics):
    from tensordg.cases.lax_1d import lax_parameters

    assert metrics["verdict_robustness"]["tolerance"] == pytest.approx(
        float(lax_parameters()["acceptance_threshold"]))


def test_no_verdict_hides_inside_the_band(metrics):
    rob = metrics["verdict_robustness"]
    assert rob["no_flips"] is True
    for row in rob["rows"]:
        assert row["robust"] is True, row
        assert row["distance_to_bar"] > row["band"]
    tightest = rob["tightest"]
    assert tightest["ratio"] == pytest.approx(
        min(r["distance_to_bar"] / r["band"] for r in rob["rows"]))


def test_certified_configuration_is_robust_and_the_pinned_leg_still_fails(metrics):
    rob = metrics["verdict_robustness"]
    assert rob["certified_leg"] == "p2.N800"
    assert rob["certified_row"]["status"] == "PASS"
    assert rob["certified_row"]["robust"] is True
    pinned = next(r for r in rob["rows"] if r["leg"] == "p3.N400")
    assert pinned["status"] == "FAIL", "the pinned ladder leg's honest FAIL must not be re-labelled"


def test_hypotheses_are_recorded_with_their_verdicts(metrics):
    verdicts = {h["id"]: h["verdict"] for h in metrics["hypotheses"]}
    assert verdicts["H-thread"] == "FALSIFIED"
    assert verdicts["H-stale-code"] == "FALSIFIED"
    assert verdicts["H-host-class"] == "SUPPORTED"


def test_doc_is_a_pure_function_of_the_artifact(metrics):
    doc = (X.OUT / "summary.md").read_text()
    assert f"{metrics['crossenv_band']['max_band']:.3e}" in doc
    assert metrics["metrics_sha256"] in doc
    assert f"{metrics['verdict_robustness']['tightest']['leg']}" in doc