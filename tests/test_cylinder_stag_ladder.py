"""CKS1998 §5.4 cylinder: the exact stagnation-pressure target and the ladder gate.

The point of these tests is that the number the cylinder case is judged against
is *derived twice*: once from the closed-form relations already in the repo
(``benchmarks.cylinder_euler_steady``) and once from the Rankine-Hugoniot
relations written out by hand in ``benchmarks.cylinder_stag_ladder``.  A defect in
either derivation fails here rather than silently moving the acceptance bar.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from benchmarks.cylinder_euler_steady import (isentropic_total_pressure, rayleigh_pitot)
from benchmarks.cylinder_stag_ladder import (_stagnation_band, _symmetry_error,
                                             independent_stagnation_target, run_case)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "cylinder_stag_ladder"
COMMITTED = ROOT / "validation" / "cylinder_steady" / "metrics_p0_6x12.json"
GAMMA = 1.4


# --------------------------------------------------------------------------- #
# 1. the acceptance target must not be a fitted or hand-typed number
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mach", [0.2, 0.38, 0.6, 0.9, 1.0, 1.5, 2.0, 3.0, 5.0])
def test_target_equals_closed_form_relations(mach: float) -> None:
    """Hand-derived RH target == ``2 (p0/p_inf - 1)/(γ M²)`` with p0 from the repo formulas."""
    total = (isentropic_total_pressure(mach) if mach < 1.0 else rayleigh_pitot(mach))
    expected = 2.0 * (total - 1.0) / (GAMMA * mach ** 2)
    got = independent_stagnation_target(mach)
    assert abs(got - expected) <= 1e-12 * max(1.0, abs(expected))


def test_rayleigh_pitot_matches_normal_shock_tables() -> None:
    """``p02/p1`` at M=3, γ=1.4 is 12.061 (normal-shock tables), not p01/p1 = 36.73."""
    p02_p1 = rayleigh_pitot(3.0)
    assert p02_p1 == pytest.approx(12.0612, rel=2e-4)
    assert isentropic_total_pressure(3.0) == pytest.approx(36.732, rel=1e-4)
    # the 5x mistake of using the isentropic total pressure across a shock
    assert independent_stagnation_target(3.0) == pytest.approx(1.7557, rel=1e-4)


def test_target_tends_to_one_in_the_incompressible_limit() -> None:
    for mach in (1e-2, 1e-3):
        assert independent_stagnation_target(mach) == pytest.approx(1.0, abs=1e-4)


def test_subsonic_stagnation_target_matches_isentropic_value() -> None:
    """M=0.38: ``Cp_stag = 2((1+0.2M²)^3.5 - 1)/(γM²)`` = 1.0368."""
    assert independent_stagnation_target(0.38) == pytest.approx(1.0368, rel=1e-3)


# --------------------------------------------------------------------------- #
# 2. the wall diagnostics must be able to detect a symmetry defect at all
# --------------------------------------------------------------------------- #
def test_symmetry_error_is_zero_for_a_symmetric_profile() -> None:
    theta = np.linspace(-np.pi, np.pi, 40, endpoint=False)
    cp = np.cos(theta)
    assert _symmetry_error(theta, cp) < 1e-12


def test_symmetry_error_detects_a_one_sided_defect() -> None:
    theta = np.linspace(-np.pi, np.pi, 40, endpoint=False)
    cp = np.cos(theta)
    broken = cp + np.where(theta > 0, 0.1, 0.0)
    # max|dCp| = 0.1, max|Cp| ~ 1.0 -> error ~ 0.1 / 1.05
    assert _symmetry_error(theta, broken) == pytest.approx(0.1 / np.max(np.abs(broken)),
                                                           rel=0.05)


def test_stagnation_band_helper_is_a_local_mean() -> None:
    theta = np.linspace(-np.pi, np.pi, 72, endpoint=False)
    cp = 1.0 + 0.0 * theta
    assert _stagnation_band(theta, cp) == pytest.approx(1.0)
    theta = np.linspace(-np.pi, np.pi, 73)      # contains theta = 0
    tilted = 1.0 - 0.5 * np.abs(theta)          # sharp peak at theta = 0
    band = _stagnation_band(theta, tilted)
    assert 0.9 < band < 1.0                     # local mean, not the global mean 0.21
    assert band < float(np.max(tilted))


# --------------------------------------------------------------------------- #
# 3. the committed m21767 cylinder artifact is a start-up transient, and the
#    driver reproduces it (attribution control C1)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not COMMITTED.exists(), reason="committed m21767 artifact absent")
def test_committed_cylinder_run_is_a_transient() -> None:
    data = json.loads(COMMITTED.read_text())
    # 35 steps / t = 0.307 on an M=3 bow-shock problem: the shock cannot exist yet
    assert data["time"] < 0.35
    assert data["steps"] < 50
    assert data["cp_stag_rel_error"] > 0.5          # the 77 % deficit reported there


@pytest.mark.skipif(not (OUT / "summary_control.json").exists(),
                    reason="control run absent (python -m benchmarks.cylinder_stag_ladder --control)")
def test_control_c1_reproduces_the_committed_number() -> None:
    summary = json.loads((OUT / "summary_control.json").read_text())
    rows = {r["tag"]: r for r in summary["rows"]}
    c1 = next(v for k, v in rows.items() if "C1_committed" in k)
    committed = json.loads(COMMITTED.read_text())
    assert c1["time"] == pytest.approx(committed["time"], abs=2e-2)
    # same configuration, same early time -> same (wrong) wall pressure
    assert c1["cp_stag"] == pytest.approx(committed["cp_stag"], abs=0.02)
    assert c1["cp_stag_rel_error"] > 0.5


@pytest.mark.skipif(not (OUT / "summary_control.json").exists(),
                    reason="control run absent")
def test_control_c3_shows_resolution_changes_the_early_time_value() -> None:
    """At the *same* early time a resolved grid is far from the committed value.

    This is what makes the committed number unusable as a resolution statement:
    it mixes a start-up transient with a 6x12 grid.
    """
    summary = json.loads((OUT / "summary_control.json").read_text())
    rows = {r["tag"]: r for r in summary["rows"]}
    c1 = next(v for k, v in rows.items() if "C1_committed" in k)
    c3 = next(v for k, v in rows.items() if "C3_resolved" in k)
    assert c3["time"] == pytest.approx(c1["time"], abs=2e-2)
    assert abs(c3["cp_stag"] - c1["cp_stag"]) > 0.5


# --------------------------------------------------------------------------- #
# 4. the ladder gate
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (OUT / "summary_ladder.json").exists(),
                    reason="ladder absent (python -m benchmarks.cylinder_stag_ladder --ladder)")
def test_ladder_rows_are_self_consistent_and_symmetric() -> None:
    summary = json.loads((OUT / "summary_ladder.json").read_text())
    assert summary["rows"], "empty ladder"
    for row in summary["rows"]:
        assert row["blew_up"] is False, row["tag"]
        assert math.isfinite(row["cp_stag"])
        recomputed = abs(row["cp_stag"] - row["cp_stag_target"]) / row["cp_stag_target"]
        assert row["cp_stag_rel_error"] == pytest.approx(recomputed, rel=1e-9, abs=1e-12)
        # y -> -y invariance of the configuration: any asymmetry is discretisation
        assert row["symmetry_error"] < 1e-8, row["tag"]


@pytest.mark.skipif(not (OUT / "summary_ladder.json").exists(),
                    reason="ladder absent")
def test_ladder_min_error_is_below_three_percent() -> None:
    """The CKS1998 §5.4 stagnation gate at the finest steady rung."""
    summary = json.loads((OUT / "summary_ladder.json").read_text())
    best = summary["best"]
    assert best is not None
    # the rungs are read once the residual has collapsed by >= 10x (plateau); a
    # full steady-state stop costs more than the local 10-min budget, so the
    # definitive steady ladder is the cluster job cj3728 and is not claimed here
    row = next(r for r in summary["rows"] if r["tag"] == best["tag"])
    assert row["residual_ratio"] < 0.1, row
    assert best["cp_stag_rel_error"] < 0.03, best


# --------------------------------------------------------------------------- #
# 5. cheap smoke run of the driver itself (kept under ~1 min)
# --------------------------------------------------------------------------- #
def test_smoke_run_is_stable_and_symmetric() -> None:
    row = run_case(8, 16, p=0, mach=3.0, r_out=5.0, growth=1.35, max_time=1.0,
                   steady_tol=1e-6, record=False)
    assert row["blew_up"] is False
    assert np.isfinite(row["cp_stag"])
    assert row["symmetry_error"] < 1e-8
    assert 0.5 < row["cp_stag"] < 3.0        # same order as the target 1.7557
    assert abs(row["mass_drift_rel"]) < 0.5  # start-up filling only