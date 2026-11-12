"""Tests for the theta--beta--Mach anchor set and the DG oblique-shock matrix.

Mission m15977.  These pin the *anchor* side of the verification (the numbers
that the DG runs are judged against) and the bookkeeping of the case matrix, so
a regression in the branch algebra cannot silently launder a solver error.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from benchmarks import oblique_shock as os
from benchmarks import oblique_shock_theta_beta as tb
from tensordg.euler_1d import GAMMA

ANCHOR_FILE = Path("benchmarks/oblique_shock/theta_beta_anchors.json")


@pytest.fixture(scope="module")
def anchors():
    return tb.build_anchors()


def test_all_limit_checks_pass(anchors):
    names = [check["name"] for check in anchors["limit_checks"]]
    assert names == [
        "theta_zero_weak_degenerates_to_mach_line",
        "theta_zero_strong_degenerates_to_normal_shock",
        "theta_max_merges_the_two_branches",
        "branch_continuity_and_ordering",
        "jump_closure_on_both_branches",
    ]
    failed = [check["name"] for check in anchors["limit_checks"]
              if check["status"] != "PASS"]
    assert not failed, failed


def test_anchor_table_covers_matrix(anchors):
    rows = anchors["table"]
    machs = {row["mach"] for row in rows}
    assert machs == set(tb.MACH_LIST) and len(machs) >= 5
    for mach in tb.MACH_LIST:
        thetas = {row["theta_deg"] for row in rows if row["mach"] == mach}
        assert len(thetas) >= 4, (mach, thetas)
        branches = {row["branch"] for row in rows if row["mach"] == mach}
        assert branches == {"weak", "strong"}
    for row in rows:
        assert row["rh_residual_max"] < 1e-12
        assert row["theta_roundtrip_deg"] == pytest.approx(row["theta_deg"], abs=1e-9)
        assert row["theta_from_vectors_deg"] == pytest.approx(row["theta_deg"], abs=1e-9)
        assert row["normal_mach"] > 1.0


def test_detachment_angles_match_textbook_values(anchors):
    # Anderson, Modern Compressible Flow, 3rd ed., ch. 4 (theta_max table).
    reference = {2.0: 22.97, 3.0: 34.07, 4.0: 38.77, 10.0: 44.4}
    for mach, expected in reference.items():
        entry = anchors["detachment"][f"{mach}"]
        assert entry["theta_max_deg"] == pytest.approx(expected, abs=0.05)


def test_strong_branch_ordering_and_limits():
    mach = 3.0
    theta_max, beta_det = tb.detachment_angle(mach)
    for theta in (0.1 * theta_max, 0.5 * theta_max, 0.9 * theta_max):
        weak = tb.beta_from_theta_branch(mach, theta, "weak")
        strong = tb.beta_from_theta_branch(mach, theta, "strong")
        assert weak < strong < 90.0
        assert os.theta_from_beta(mach, weak) == pytest.approx(theta, abs=1e-9)
        assert os.theta_from_beta(mach, strong) == pytest.approx(theta, abs=1e-9)
    # theta -> 0 on the strong branch is the normal shock (beta -> 90)
    assert tb.beta_from_theta_branch(mach, 1e-6, "strong") == pytest.approx(90.0, abs=1e-2)
    # theta -> 0 on the weak branch is the Mach line
    mu = math.degrees(math.asin(1.0 / mach))
    assert tb.beta_from_theta_branch(mach, 1e-6, "weak") == pytest.approx(mu, abs=1e-4)


def test_detached_deflection_is_rejected_on_both_branches():
    theta_max, _ = tb.detachment_angle(2.0)
    for branch in ("weak", "strong"):
        with pytest.raises(ValueError):
            tb.beta_from_theta_branch(2.0, theta_max + 1e-3, branch)
        with pytest.raises(ValueError):
            tb.beta_from_theta_branch(2.0, 0.0, branch)


def test_gci_recovers_a_known_second_order_rate():
    # error ~ C h^2 with h halving: values approach 10.0 quadratically
    coarse, medium, fine = 10.0 - 0.8, 10.0 - 0.2, 10.0 - 0.05
    result = tb.gci_three_level(fine, medium, coarse)
    assert result["order_estimated"] is True
    assert result["observed_order"] == pytest.approx(2.0, abs=1e-6)
    assert result["richardson_limit"] == pytest.approx(10.0, abs=1e-9)
    assert 0.0 < result["gci_fine_pct"] < 2.0


def test_gci_flags_a_non_decaying_oscillatory_triplet():
    # no decay at all (the two differences are equal in size) -> observed order 0,
    # which is outside the accepted band, so the estimate must be flagged and the
    # theoretical order used instead of silently reporting a meaningless rate.
    result = tb.gci_three_level(1.0, 1.000001, 1.0)
    assert result["order_estimated"] is False
    assert result["order_used"] == 1.0
    assert result["gci_fine_pct"] >= 0.0


def test_matrix_case_ids_are_unique_and_branches_present():
    cases = tb.build_matrix(ncells=16, final_time=0.01)
    ids = [case.case_id for case in cases]
    assert len(ids) == len(set(ids))
    weak = {(c.mach, c.theta_deg, c.degree) for c in cases if c.branch == "weak"}
    assert len({m for m, _, _ in weak}) >= 5
    assert len({(m, t) for m, t, _ in weak}) >= 5 * 4
    assert {c.degree for c in cases} == {1, 2}
    assert any(c.branch == "strong" for c in cases)


def test_one_small_dg_case_reproduces_the_anchor(tmp_path):
    case = tb.MatrixCase(3.0, 10.0, "weak", 1, ncells=16, final_time=0.01)
    record = tb.run_matrix_case(case)
    # primary metric: robust gradient-centroid fit; the mid-level crossing from
    # benchmarks.oblique_shock is reported alongside as the diagnostic estimator
    assert record["beta_dev_centroid_pct"] < tb.BETA_GATE_PCT
    assert record["beta_dev_pct"] is not None
    assert record["rho_dev_pct"] < tb.JUMP_GATE_PCT
    assert record["p_dev_pct"] < tb.JUMP_GATE_PCT
    assert record["verdict"] == "PASS"
    rows = tb.run_matrix([case], output_dir=tmp_path)
    assert len(rows) == 1
    with open(tmp_path / "case_matrix.csv") as handle:
        parsed = list(csv.DictReader(handle))
    assert parsed[0]["case_id"] == case.case_id
    assert parsed[0]["verdict"] == "PASS"


@pytest.mark.skipif(not ANCHOR_FILE.exists(),
                    reason="anchor artifact not generated yet")
def test_checked_in_anchor_file_matches_the_generator(anchors):
    on_disk = json.loads(ANCHOR_FILE.read_text())
    assert on_disk["anchor_id"] == tb.ANCHOR_ID
    assert on_disk["gamma"] == GAMMA
    assert len(on_disk["table"]) == len(anchors["table"])
    for stored, fresh in zip(on_disk["table"], anchors["table"]):
        assert stored["mach"] == fresh["mach"]
        assert stored["theta_deg"] == fresh["theta_deg"]
        assert stored["branch"] == fresh["branch"]
        assert stored["beta_deg"] == pytest.approx(fresh["beta_deg"], abs=1e-9)
        assert stored["rho_ratio"] == pytest.approx(fresh["rho_ratio"], rel=1e-12)
        assert stored["p_ratio"] == pytest.approx(fresh["p_ratio"], rel=1e-12)
    assert [c["status"] for c in on_disk["limit_checks"]] == \
           [c["status"] for c in anchors["limit_checks"]]

def test_centroid_estimator_beats_the_crossing_estimator_on_a_shallow_front():
    """The published failure mode: mid-level crossings on a shallow, smeared front.

    At M = 6, theta = 15 the mid-level crossing fit is biased by the one-cell
    staircase of the front (16 % angle error) while the field itself is
    preserved to 3.6 %; the gradient-centroid fit stays inside its gate.
    """
    case = tb.MatrixCase(6.0, 15.0, "weak", 2, ncells=32, final_time=0.05)
    record = tb.run_matrix_case(case)
    assert record["beta_dev_pct"] > 5.0                  # crossing estimator biased
    assert record["beta_dev_centroid_pct"] < tb.BETA_GATE_PCT
    assert record["preserve_l1"] < 0.05                   # field itself preserved


def test_merge_parts_deduplicates_and_orders(tmp_path):
    header = ",".join(tb.CSV_FIELDS)
    row = {key: "" for key in tb.CSV_FIELDS}
    row.update({"case_id": "M3_th10_weak_P1", "mach": "3", "theta_deg": "10",
                "branch": "weak", "degree": "1"})
    other = dict(row, case_id="M1.5_th2_weak_P2", mach="1.5", theta_deg="2",
                 degree="2")
    (tmp_path / "part_a.csv").write_text(
        header + "\n" + ",".join(str(row[k]) for k in tb.CSV_FIELDS) + "\n")
    (tmp_path / "part_b.csv").write_text(
        header + "\n" + ",".join(str(other[k]) for k in tb.CSV_FIELDS) + "\n"
        + ",".join(str(row[k]) for k in tb.CSV_FIELDS) + "\n")
    merged = tb.merge_parts(tmp_path)
    with open(merged) as handle:
        rows = list(csv.DictReader(handle))
    assert [r["case_id"] for r in rows] == ["M1.5_th2_weak_P2", "M3_th10_weak_P1"]
