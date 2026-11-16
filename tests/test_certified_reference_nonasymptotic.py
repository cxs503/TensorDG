"""Regression + live recompute for `docs/CERTIFIED-REFERENCE-NONASYMPTOTIC-BAND.md`.

The document is generated from `artifacts/certify_nonasymptotic_band/results.json`;
these tests keep the two from drifting and re-run one small ladder so the
qualitative claim is reproduced by code, not only read out of an artifact.
"""
import json
from pathlib import Path

import pytest

from tools.certify_nonasymptotic_band import CASES, measure_case

RESULTS = Path("artifacts/certify_nonasymptotic_band/results.json")


@pytest.fixture(scope="module")
def payload():
    if not RESULTS.exists():  # pragma: no cover - artifact is committed
        pytest.skip(f"{RESULTS} not present")
    return json.loads(RESULTS.read_text())


def _ladders(case):
    for key in ("three_level", "four_level"):
        m = case.get(key)
        if m:
            yield key, m


def test_every_ladder_is_internally_consistent(payload):
    assert payload["tau_percent"] == 3.0
    for case in payload["cases"]:
        for key, m in _ladders(case):
            if m["band_covers_truth"] is not None:
                # the boolean is exactly the inequality, not a parallel claim
                assert m["band_covers_truth"] == (
                    m["err_certified_percent"] <= m["band_percent"])
                assert m["true_over_band"] == pytest.approx(
                    m["err_certified_percent"] / m["band_percent"], rel=1e-9)
            if m["band_share_of_gate"] is not None:
                # X3: gate validity is the engine's own U/tau <= 1/3 rule
                assert m["gate_usable"] == (m["band_share_of_gate"] <= 1.0 / 3.0)
            assert 0.0 < m["err_raw_finest_percent"]


def test_summary_matches_the_ladders(payload):
    summary = payload["summary"]
    in_regime = [m for c in payload["cases"] for _, m in _ladders(c)
                 if m["true_over_band"] is not None
                 and m["observed_order"] is not None and m["observed_order"] <= 1.5]
    assert summary["n_in_regime"] == len(in_regime)
    if in_regime:
        worst = max(m["true_over_band"] for m in in_regime)
        assert summary["worst_true_over_band_in_regime"] == pytest.approx(worst, rel=1e-9)
        assert summary["verdict"] == (
            "COVERED_ALL" if worst <= 1.0 else "UNDER_COVERS")


def test_the_headline_finding_is_the_one_documented(payload):
    """p < 1 limiter ladder: the band covers the truth, yet the gate is consumed."""
    case = next(c for c in payload["cases"] if c["case"] == "adv_square_tvb_P1")
    m = case["three_level"]
    assert m["verdict"] == "WITHHELD_BAND_CONSUMES_GATE"
    assert m["observed_order"] < 1.0, "must be in the slow-convergence regime"
    assert m["true_over_band"] <= 1.0, "band must still cover the true error"
    assert m["band_share_of_gate"] > 1.0 / 3.0, "and yet the band eats the 3% gate"
    assert m["richardson_gain"] > 1.0, "extrapolation still improves on the raw level"


def test_live_small_ladder_reproduces_the_slow_regime(tmp_path):
    """Recompute a small square-wave ladder through the production kernel."""
    case = dict(CASES[0])
    case["ladder"] = [50, 100, 200]
    row = measure_case(case, outdir=tmp_path)
    m = row["three_level"]
    assert m["observed_order"] is not None
    assert m["observed_order"] < 1.5, "the limiter keeps the order in the slow regime"
    if m["true_over_band"] is not None:
        assert m["true_over_band"] <= 1.0, "band must cover the measured true error"
