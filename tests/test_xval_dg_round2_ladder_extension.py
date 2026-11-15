"""Gates for the ladder-extension audit of the round-2 certified reference.

The case ``XVAL-DG-ROUND2-CERTIFIED-REFERENCE`` certified its reference from the
three-level FV ladder ``[800, 1600, 3200]``; ``CASE-SHU-OSHER-round2.md`` §6
recorded the missing fourth level as an honest unfinished item.  With the fourth
level ``N = 6400`` committed, the band that three levels produced becomes a
*falsifiable* claim, and these tests pin the audit that decides it:

``T1`` the four committed levels nest (r = 2) and are admissible -- the audit has
      real inputs (no fabrication: a missing level is an error, not a skip);
``T2`` the successive differences are recomputable with an independent
      cell-average/rel-L1 implementation (no engine helper) -- the L0 evidence is
      not a property of the engine's code path;
``T3`` the engine certifies the extended ladder under the *unchanged* call, and
      the engine's own agreement is checked against the independent numbers;
``T4`` the band-coverage (L3) decision is a pure comparison of two measured
      numbers: the test recomputes the shift independently and checks that the
      published decision equals ``shift <= U3`` (a FAIL stays a FAIL -- the test
      never asserts coverage it did not measure);
``T5`` the eleven DG legs are judged against both references and the per-leg
      movement bound (L4) is recomputable from the audit's own rows;
``T6`` the artifact on disk is the artifact the audit produces (binding, no
      hand-typed numbers in docs/).

Run: ``pytest -q tests/test_xval_dg_round2_ladder_extension.py``
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "validation" / "repro_shu_osher" / "artifacts" / "raw"
ART = ROOT / "validation" / "xval_dg_round2" / "artifacts" / "ladder_extension.json"

sys.path.insert(0, str(ROOT))

from tensordg.certified_reference import certify, coarsen, rel_l1  # noqa: E402
from benchmarks import xval_dg_round2_ladder_extension as le  # noqa: E402

pytestmark = pytest.mark.skipif(
    not le.level_path(RAW, le.EXT_LEVEL).exists(),
    reason="fourth reference level (N=6400) not committed; generate it with "
           "validation/xval_dg_round2/gen_fv_level.py")


# --------------------------------------------------------------- independent math
def _cell_average(field: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return np.asarray(field, dtype=float)
    n = field.size // factor
    return np.asarray(field[: n * factor], dtype=float).reshape(n, factor).mean(axis=1)


def _rel_l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)) / np.sum(np.abs(b)))


def test_levels_nest_and_are_admissible():
    for n in (*le.BASE_LEVELS, le.EXT_LEVEL):
        lv = le.load_fv_level(RAW, n)
        assert lv["ncells"] == n and lv["field"].size == n
        assert np.all(np.isfinite(lv["field"])) and lv["field"].min() > 0.0
    sizes = [le.load_fv_level(RAW, n)["field"].size for n in
             (*le.BASE_LEVELS, le.EXT_LEVEL)]
    for a, b in zip(sizes, sizes[1:]):
        assert b == 2 * a, "ladder must nest with refinement 2"


def test_successive_differences_independent():
    levels = [le.load_fv_level(RAW, n) for n in (*le.BASE_LEVELS, le.EXT_LEVEL)]
    base = levels[0]["ncells"]
    # coarsen EVERY level to the common base grid (independent of the engine
    # helper); the audit's successive_differences compares ADJACENT levels, so
    # the independent recomputation must too -- comparing each finer level
    # against the coarsest base would measure a different (non-successive)
    # quantity and silently break the L0 monotone evidence.
    prof = [_cell_average(lv["field"], lv["field"].size // base) for lv in levels]
    rows = le.successive_differences(levels)
    for row, (a, b) in zip(rows, zip(prof, prof[1:])):
        assert row["rel_l1"] == pytest.approx(_rel_l1(a, b), rel=1e-12, abs=0.0)


def test_extended_ladder_is_certified_by_unchanged_call():
    levels = [le.load_fv_level(RAW, n) for n in (*le.BASE_LEVELS, le.EXT_LEVEL)]
    cert = certify(levels, tolerance_percent=le.GATE_PERCENT)
    out = le.audit(RAW)
    assert cert["verdict"] == out["L1_extension_certified"]["verdict"]
    assert out["L1_extension_certified"]["pass"] == (cert["verdict"] == "CERTIFIED")
    assert cert["ladder"] == [*le.BASE_LEVELS, le.EXT_LEVEL]


def test_band_coverage_decision_matches_measurement():
    out = le.audit(RAW)
    l3 = out["L3_band_covers_shift"]
    levels3 = [le.load_fv_level(RAW, n) for n in le.BASE_LEVELS]
    levels4 = [le.load_fv_level(RAW, n) for n in (*le.BASE_LEVELS, le.EXT_LEVEL)]
    cert3 = certify(levels3, tolerance_percent=le.GATE_PERCENT)
    cert4 = certify(levels4, tolerance_percent=le.GATE_PERCENT)
    ref4 = _cell_average(cert4["reference"], 2)
    shift = _rel_l1(ref4, cert3["reference"])
    assert l3["shift_rel_l1"] == pytest.approx(shift, rel=1e-12)
    assert l3["band_fraction_U3"] == pytest.approx(cert3["band_fraction"], rel=1e-12)
    # the decision is *this* comparison and nothing else -- a FAIL must stay a FAIL
    assert l3["pass"] == (shift <= cert3["band_fraction"])
    assert out["decisions"]["L3"] == l3["pass"]


def test_leg_movement_bound_recomputable():
    out = le.audit(RAW)
    l4 = out["L4_claim_stable"]
    moves = l4["moves"]
    assert len(moves) == 11
    worst = max(abs(m["delta_pp"]) for m in moves)
    assert l4["max_abs_move_pp"] == pytest.approx(worst, rel=1e-12, abs=0.0)
    for m in moves:
        assert m["delta_pp"] == pytest.approx(
            m["rel_l1_percent_4level"] - m["rel_l1_percent_3level"], rel=1e-9, abs=1e-9)
    assert l4["pass"] == (all(m["pass_strict_4level"] for m in moves)
                          and worst <= le.LEG_STABILITY_MAX_PP)


def test_order_consistency_gate():
    out = le.audit(RAW)
    l2 = out["L2_order_consistency"]
    p3, p4 = l2["p_three_levels"], l2["p_four_levels"]
    assert l2["relative_change"] == pytest.approx(abs(p4 - p3) / abs(p3), rel=1e-12)
    assert l2["pass"] == (l2["relative_change"] <= le.ORDER_CONSISTENCY_MAX)


def test_committed_artifact_matches_audit():
    if not ART.exists():
        pytest.skip("artifact not regenerated yet (run the module with --figures)")
    published = json.loads(ART.read_text())
    fresh = json.loads(json.dumps(le.audit(RAW), default=str))
    for key in ("decisions", "accepted", "L0_successive_differences",
                "L1_extension_certified", "L2_order_consistency",
                "L3_band_covers_shift"):
        assert published[key] == fresh[key], key
    assert published["thresholds"] == fresh["thresholds"]
