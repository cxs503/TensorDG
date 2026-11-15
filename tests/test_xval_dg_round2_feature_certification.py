"""Tests for the feature-level certification audit of CASE-SHU-OSHER round 2.

Two layers:

1. **The measurement apparatus**, asserted against an *analytic* profile -- a
   sine of known amplitude inside the statistics window, whose ``amplitude``,
   ``rms`` and ``l1`` are known in closed form.  If the extractor is wrong, a
   PASS here is impossible.
2. **The audit's conclusions**, asserted only as *internal consistency* between
   the shipped artifact and a fresh recomputation from the committed raw
   profiles, plus the structural claims the report makes (which feature is
   certifiable, and that a global-L1 PASS leg fails the feature gate).  The
   numbers themselves live in ``validation/xval_dg_round2/artifacts/
   feature_certification/`` and are recomputed by the driver.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from benchmarks.xval_dg_round2_feature_certification import (
    GATE_PERCENT,
    OUTDIR,
    features,
    leg_path,
    load_profile,
    run_audit,
    wave_window,
)

ARTIFACT = OUTDIR / "feature_certification_audit.json"


def synthetic_profile(n: int, amplitude: float = 0.4, k: float = 5.0,
                      plateau: float | None = None):
    """A sine of known amplitude sampled at cell centres on the round-2 grid."""
    x_lo, x_hi, plat = wave_window()
    plateau = plat if plateau is None else plateau
    dx = 10.0 / n
    x = -5.0 + (np.arange(n) + 0.5) * dx
    return x, plateau + amplitude * np.sin(k * x)


# --------------------------------------------------------------- apparatus
def test_analytic_amplitude_and_integrals_are_recovered():
    x, rho = synthetic_profile(6400, amplitude=0.4, k=5.0)
    f = features(x, rho)
    # amplitude of a sine is exactly the amplitude; rms = A/sqrt2; l1 = 2A/pi
    assert f["amplitude"] == pytest.approx(0.4, rel=2e-3)
    assert f["rms"] == pytest.approx(0.4 / math.sqrt(2.0), rel=2e-2)
    assert f["l1"] == pytest.approx(2.0 * 0.4 / math.pi, rel=2e-2)
    assert f["crest"] == pytest.approx(f["rho_plateau"] + 0.4, rel=2e-3)


def test_features_are_grid_invariant_for_a_resolved_profile():
    """Doubling the grid must not move the functionals (sampling, not physics)."""
    x1, r1 = synthetic_profile(800)
    x2, r2 = synthetic_profile(1600)
    f1, f2 = features(x1, r1), features(x2, r2)
    for name in ("amplitude", "rms", "l1"):
        assert f1[name] == pytest.approx(f2[name], rel=2e-2), name


def test_window_is_bounded_by_the_analytic_anchors():
    from tensordg.cases.shock_entropy import rankine_hugoniot_anchor

    rh = rankine_hugoniot_anchor()
    x_lo, x_hi, plateau = wave_window()
    assert x_lo > float(rh["quiet_zone_head_analytic"])       # inset inside the head
    assert x_hi < float(rh["shock_position_analytic"])        # inset before the shock
    assert plateau == pytest.approx(float(rh["post_shock_plateau"]["rho"]))


def test_a_grid_too_coarse_for_the_window_is_refused():
    x = np.array([-5.0, -2.0, 1.0, 4.0])
    rho = np.ones_like(x)
    with pytest.raises(ValueError, match="too coarse"):
        features(x, rho)


def test_shape_mismatch_is_refused():
    with pytest.raises(ValueError):
        features(np.zeros(10), np.zeros(11))


# --------------------------------------------------------------- audit
@pytest.fixture(scope="module")
def audit() -> dict:
    return run_audit()


def test_committed_artifact_matches_a_fresh_recomputation(audit):
    if not ARTIFACT.exists():
        pytest.skip("shipped artifact not present")
    shipped = json.loads(ARTIFACT.read_text())
    for name, st in audit["features"].items():
        a = shipped["features"][name]["weno5"]["certified_value"]
        b = st["weno5"]["certified_value"]
        if a is None or b is None:
            assert a is None and b is None, name
            continue
        assert a == pytest.approx(b, rel=1e-9), name
    assert shipped["usable_features"] == audit["usable_features"]
    assert (len(shipped["G4_counterexamples"]) == len(audit["G4_counterexamples"]))


def test_only_the_amplitude_functional_survives_both_families(audit):
    """G0 is a two-family rule: one tight band is not evidence."""
    assert audit["usable_features"] == ["amplitude"]
    amp = audit["features"]["amplitude"]
    assert amp["G0_certifiable"] and amp["G1_gate_usable"]
    assert amp["weno5"]["verdict"] == "CERTIFIED"
    assert amp["weno7"]["verdict"] == "CERTIFIED"
    # both families agree well inside their combined band
    assert amp["G2_cross_family"]["covered"] is True
    # and the band is small enough for a 3 % gate to be non-vacuous
    assert amp["weno5"]["band_percent"] / GATE_PERCENT <= audit["u_share_max"]


def test_uncertifiable_functionals_are_refused_not_averaged(audit):
    for name in ("crest", "rms", "l1", "peak_x"):
        st = audit["features"][name]
        assert st["G0_certifiable"] is False, name
        # every one of them is refused because *some* family's ladder is not
        # asymptotic -- never because the engine liked one family more
        verdicts = (st["weno5"]["verdict"], st["weno7"]["verdict"])
        assert any(v.startswith("WITHHELD") for v in verdicts), (name, verdicts)


def test_profile_gate_is_insufficient_at_feature_level(audit):
    """The finding: a leg that PASSes the 3 % L1 gate can still lose the wave train."""
    assert audit["G4_profile_gate_sufficient"] is False
    assert len(audit["G4_counterexamples"]) == len(audit["legs"]) == 11
    for leg in audit["legs"]:
        assert leg["round2_profile_decision"] == "PASS"
        assert leg["feature_judgements"]["amplitude"]["decision"] == "FAIL"
        assert leg["feature_judgements"]["amplitude"]["error_percent"] > GATE_PERCENT


def test_deficit_shrinks_with_resolution(audit):
    """A scheme effect would not care about N; a resolution effect must."""
    errs = [(leg["ncells"], leg["feature_judgements"]["amplitude"]["error_percent"])
            for leg in audit["legs"] if leg["cfl"] == 0.4 and leg["degree"] == 2]
    by_n = dict()
    for n, e in errs:
        by_n.setdefault(n, []).append(e)
    ns = sorted(by_n)
    means = [float(np.mean(by_n[n])) for n in ns]
    assert means[0] > means[1] > means[2], (ns, means)


def test_same_grid_comparison_attributes_the_deficit_to_the_scheme(audit):
    """At N=800 the reference family is within 3 %; the DG legs are not -- so the
    deficit is not a resolution artefact of the statistics window."""
    rows = [r for r in audit["same_grid_cross_scheme"] if r["ncells"] == 800]
    assert rows, "same-grid comparison missing"
    row = rows[0]
    assert row["weno5_error_percent"] < GATE_PERCENT
    assert row["weno7_error_percent"] < row["weno5_error_percent"]
    for deg in (1, 2, 3):
        assert row[f"dg_p{deg}_error_percent"] > 5.0 * row["weno5_error_percent"]


def test_cfl_is_not_the_cause(audit):
    """The three CFL variants of DG P2 N=400 agree to well under the deficit."""
    cfl_legs = [leg for leg in audit["legs"] if leg["ncells"] == 400
                and leg["degree"] == 2]
    assert len(cfl_legs) == 3
    errs = [leg["feature_judgements"]["amplitude"]["error_percent"] for leg in cfl_legs]
    assert max(errs) - min(errs) < 0.5        # percent
    assert min(errs) > GATE_PERCENT


def test_leg_paths_point_at_committed_profiles():
    for degree, ncells, cfl in ((1, 200, 0.4), (3, 800, 0.4), (2, 400, 0.1)):
        path = leg_path(ncells, degree, cfl)
        assert path.exists(), path
        x, rho = load_profile(path)
        assert x.shape == rho.shape and rho.size == ncells