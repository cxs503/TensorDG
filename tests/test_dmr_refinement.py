"""Regression gates for the DMR mesh-refinement ladder (anchor REPRO-DG-CKS1998).

These tests never re-run the solver: they check that the committed artifacts
(``validation/dmr/refinement/``) are internally consistent, that every gate
verdict follows from its own evidence, and that the key measurement can be
recomputed from the stored raw profile.  That is what makes the verdict file a
measurement instead of a claim.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "validation" / "dmr" / "refinement"
GATES = json.loads((ROOT / "gates.json").read_text())
SUMMARY = json.loads((ROOT / "summary.json").read_text())
RUNS = [json.loads(p.read_text()) for p in sorted(ROOT.glob("*/metrics.json"))]

POST_RHO = 8.0
POST_P = 116.5


def p1_runs():
    rs = [r for r in RUNS if r["degree"] == 1]
    return sorted(rs, key=lambda r: r["grid"][0])


def test_ladder_covers_the_pre_registered_protocol():
    planned = [tuple(g) for g in GATES["protocol"]["ladder_P1"]]
    got = [tuple(r["grid"]) for r in p1_runs()]
    assert got == planned, f"ladder {got} != pre-registered {planned}"
    for r in RUNS:
        nx, ny = r["grid"]
        assert r["dx"] == pytest.approx(4.0 / nx)
        assert r["steps"] > 0 and r["wall_seconds"] > 0.0
        assert r["final_time"] == pytest.approx(0.02)


def test_rankine_hugoniot_residuals_are_machine_precision():
    for r in RUNS:
        for name, value in r["rh_residuals"].items():
            if name == "shock_speed":
                assert value == pytest.approx(10.0, rel=1e-10)
            else:
                assert abs(value) < 1e-12, (r["grid"], name, value)


def test_front_position_is_recomputed_from_the_stored_profile():
    """The headline number must be a function of the raw profile, not a note."""
    def crossing(xc, prof, level):
        diff = np.asarray(prof) - level
        for i in range(len(diff) - 1, 0, -1):
            if diff[i] < 0.0 <= diff[i - 1]:
                w = diff[i - 1] - diff[i]
                frac = diff[i - 1] / w if w > 0 else 0.0
                return 0.5 * (xc[i] + xc[i - 1]) + (frac - 0.5) * (xc[i] - xc[i - 1])
        return None

    for r in p1_runs():
        xc = np.asarray(r["profile_x"])
        rho = np.asarray(r["profile_rho"])
        assert len(xc) == r["grid"][0]
        assert np.all(np.diff(xc) > 0)
        # the profile row is one of the rows reported in front_x
        key = min(r["front_x"], key=lambda k: abs(float(k) - r["profile_y"]))
        assert abs(float(key) - r["profile_y"]) < 1e-12
        level = 0.5 * (1.4 + POST_RHO)
        recomputed = crossing(xc, rho, level)
        assert recomputed is not None
        assert recomputed == pytest.approx(r["front_x"][key], abs=1e-9)


def test_two_plateau_windows_separate_the_mechanisms():
    """The interior window and the near-top window must disagree strongly,
    otherwise the 'top boundary' diagnosis is vacuous."""
    for r in p1_runs():
        mid = abs(r["plateau_rel_error"]["mid"]["rho"])
        top = abs(r["plateau_rel_error"]["top"]["rho"])
        assert top > 5.0 * mid, (r["grid"], mid, top)


def test_top_row_front_error_is_the_outlier():
    for r in p1_runs():
        cells = {float(k): v for k, v in r["front_error_cells"].items()}
        interior = max(abs(v) for k, v in cells.items() if 0.45 <= k <= 0.85)
        top = max(abs(v) for k, v in cells.items() if k >= 0.88)
        assert top >= 2.0 * interior, (r["grid"], interior, top)


def _verdict_from_evidence(gid, ev):
    """Independent re-implementation of the gate predicates (no import)."""
    p1 = p1_runs()
    coarse, fine = p1[0], p1[-1]
    if gid == "G1":
        return all(abs(v["rho"]) <= 0.03 and abs(v["p"]) <= 0.03 for v in ev.values())
    if gid == "G2":
        return ev["fine"]["abs_rel"] < ev["coarse"]["abs_rel"]
    if gid == "G3":
        return max(ev.values()) <= 2.0
    if gid == "G4":
        return ev["fine"]["abs_phys"] <= 0.5 * ev["coarse"]["abs_phys"]
    if gid == "G5":
        return 0.5 <= ev["ratio_finest_over_coarsest"] <= 2.0
    if gid == "G6":
        e_c = abs(ev[coarse["grid"][0]]["rel_error_rho"])
        e_f = abs(ev[fine["grid"][0]]["rel_error_rho"])
        return e_f < e_c and e_f <= 0.03
    if gid == "G7":
        return ev["rate"] is not None and ev["rate"] >= 0.5
    if gid == "G8":
        return (ev["coarse"]["offset_cells"] < ev["coarse"]["smear_width_cells"]
                and ev["fine"]["offset_cells"] > ev["fine"]["smear_width_cells"])
    raise AssertionError(f"unhandled gate {gid}")


def test_every_gate_verdict_follows_from_its_own_evidence():
    gates = SUMMARY["gates"]
    assert set(gates) == {g["id"] for g in GATES["gates"]}
    for gid, g in gates.items():
        expected = _verdict_from_evidence(gid, g["evidence"])
        assert g["verdict"] == ("PASS" if expected else "FAIL"), gid


def test_gate_evidence_matches_the_run_metrics():
    """Recompute the recorded 64x16 interior plateau error from metrics.json."""
    coarse = p1_runs()[0]
    ev = SUMMARY["gates"]["G1"]["evidence"][str(coarse["grid"][0])]
    assert ev["rho"] == pytest.approx(coarse["plateau_rel_error"]["mid"]["rho"], abs=1e-12)
    assert ev["p"] == pytest.approx(coarse["plateau_rel_error"]["mid"]["p"], abs=1e-12)


def test_summary_markdown_lists_every_gate_with_its_verdict():
    md = (ROOT / "summary.md").read_text()
    for gid, g in SUMMARY["gates"].items():
        assert f"| {gid} | **{g['verdict']}** |" in md


def test_sample_offsets_are_reported_in_cells():
    """G6/G8 depend on the offset being expressed in cells; pin that."""
    for r in p1_runs():
        for name, s in r["samples_behind_front"].items():
            assert s["offset_cells"] == pytest.approx(s["offset"] / r["dx"])
            assert s["offset_cells"] > 0.0