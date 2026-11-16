"""Contract tests for the Sod DG-P2 gate positivity attribution.

The artifact must be re-adjudicable offline: the attribution is recomputed from
the stored per-field errors, the published number must stay untouched, and the
verdict must follow the pre-registered rule rather than a label.

A missing artifact reads as ``skip`` (not run here), never as green.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from validation.cks1998_dual_instrument.dg_p2_positivity_attribution import (
    ARTIFACT,
    FIELDS,
    FINGERPRINT_TOL,
    PUBLISHED,
    TOLERANCE,
    adjudicate,
)

ROOT = Path(__file__).resolve().parents[1]


def _artifact():
    if not ARTIFACT.exists():
        pytest.skip(f"attribution artifact not committed yet ({ARTIFACT}); reproduce with "
                    "`python3 -m validation.cks1998_dual_instrument."
                    "dg_p2_positivity_attribution`")
    return json.loads(ARTIFACT.read_text())


def test_attribution_is_recomputed_from_the_stored_errors():
    stored = _artifact()
    rebuilt = adjudicate([dict(c) for c in stored["candidates"]])
    assert rebuilt["status"] == stored["attribution"]["status"]
    assert rebuilt["matching_candidates"] == stored["attribution"]["matching_candidates"]
    # the rule is the verdict: exactly one match -> attributed, none -> unattributed
    matches = rebuilt["matching_candidates"]
    expected = {0: "unattributed", 1: "attributed"}.get(len(matches), "ambiguous")
    assert stored["attribution"]["status"] == expected


def test_worst_fields_are_the_maxima_of_the_stored_rows():
    stored = _artifact()
    for candidate in stored["candidates"]:
        m = candidate["measurement"]
        assert m["worst_legacy"] == pytest.approx(
            max(m["fields"][f]["legacy_rel_l2"] for f in FIELDS))
        assert m["worst_exact"] == pytest.approx(
            max(m["fields"][f]["exact_rel_l2"] for f in FIELDS))
        assert m["worst_field_legacy"] == max(
            FIELDS, key=lambda f: m["fields"][f]["legacy_rel_l2"])
        assert m["worst_field_exact"] == max(
            FIELDS, key=lambda f: m["fields"][f]["exact_rel_l2"])
        # verdicts are recomputed from the numbers, not read off a label
        for field in FIELDS:
            row = m["fields"][field]
            assert row["pass_legacy"] == (row["legacy_rel_l2"] < TOLERANCE)
            assert row["pass_exact"] == (row["exact_rel_l2"] < TOLERANCE)
            assert row["margin_exact_pp"] == pytest.approx(
                100.0 * (TOLERANCE - row["exact_rel_l2"]))


def test_published_number_and_tolerance_are_not_re_toleranced():
    stored = _artifact()
    assert stored["tolerance"] == pytest.approx(0.03)
    assert stored["published"]["rel_l2"] == pytest.approx(PUBLISHED["rel_l2"])
    assert stored["attribution"]["fingerprint_tolerance_pp"] == pytest.approx(
        100.0 * FINGERPRINT_TOL)


def test_every_candidate_declares_the_numeric_config_it_ran():
    """A reading without its positivity/stage policy is not evidence."""
    stored = _artifact()
    assert len(stored["candidates"]) >= 2
    for candidate in stored["candidates"]:
        config = candidate["config"]
        for key in ("ncells", "degree", "cfl", "final_time", "limiter",
                    "tvb_constant", "limiter_stage", "positivity"):
            assert key in config, (candidate["id"], key)
        assert candidate["steps"] > 0
    ids = [c["id"] for c in stored["candidates"]]
    assert len(set(ids)) == len(ids)
    # the registered reading and the historic policy must both be present: the
    # whole point is the comparison between them
    roles = {c["role"] for c in stored["candidates"]}
    assert {"registered", "historic"} <= roles


def test_measuring_the_same_number_two_ways_is_not_a_free_choice():
    """Instrument delta must stay far below the discrepancy being attributed.

    If the legacy and exact instruments ever disagree by an amount comparable to
    the 1.47 pp gap, the attribution would be measuring the instrument instead of
    the code state, and this test would go red.
    """
    stored = _artifact()
    for candidate in stored["candidates"]:
        for field in FIELDS:
            row = candidate["measurement"]["fields"][field]
            assert abs(row["delta_pp"]) < 1.0, (candidate["id"], field, row["delta_pp"])