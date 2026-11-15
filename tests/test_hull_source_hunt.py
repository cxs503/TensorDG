"""Offline tests for the hull-geometry source hunt (GAP-CAD-T1-SOURCE-HUNT).

Run: python3 -m pytest tests/test_hull_source_hunt.py -q
No network: every case uses a synthetic record or the committed artefacts.
"""
import json
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hull_source_hunt as h  # noqa: E402

ART = ROOT / "artifacts" / "baseline_fleet_source_hunt"


def _iges_pd(records, units="1HM"):
    """Build a minimal IGES file whose P section holds *records* (free format)."""
    g = ",,5Htest,1Hf,1Hsys,1Hpre,32,38,6,308,14,5Htest,1.0,6," + units + ",32767,32.767,1Hnow,"
    lines = []
    lines.append("test".ljust(72) + "S      1")
    gtxt = g.ljust(72)
    lines.append(gtxt[:72] + "G      1")
    for i, rec in enumerate(records, 1):
        data = rec + ";"
        while data:
            chunk, data = data[:64], data[64:]
            lines.append(chunk.ljust(64) + "       1" + "P" + str(i).rjust(7))
    lines.append("".ljust(72) + "T      1")
    return "\n".join(lines)


def test_parse_number_handles_units_and_group_spaces():
    assert h.parse_number("8507 ton") == 8507.0
    assert h.parse_number("7.27 86") == pytest.approx(7.2786)
    assert h.parse_number("52,030") == 52030.0
    assert h.parse_number("NA") is None
    assert h.parse_number("0.6505") == pytest.approx(0.6505)


def test_iges_bbox_free_format_cartesian_points():
    text = _iges_pd(["116,0.0,0.0,0.0", "116,10.0,2.0,-1.0"])
    bb = h.iges_bbox(text)
    assert bb["points"] == 2
    assert bb["units_name"] == "M"
    assert bb["span_m"]["x"] == pytest.approx(10.0)
    assert bb["span_m"]["y"] == pytest.approx(2.0)
    assert bb["span_m"]["z"] == pytest.approx(1.0)


def test_iges_bbox_millimetre_units_are_scaled_to_metres():
    text = _iges_pd(["116,0,0,0", "116,1000,500,0"], units="1HMM")
    bb = h.iges_bbox(text)
    assert bb["units_name"] == "MM"
    assert bb["span_m"]["x"] == pytest.approx(1.0)
    assert bb["span_m"]["y"] == pytest.approx(0.5)


def test_iges_bbox_nurbs_surface_uses_control_points():
    # type 128, K1=K2=1, M=N=1: knots 4+4, weights 4, then 4 control points
    rec = ("128,1,1,1,1,0,0,1,0,0," + ",".join(["0.0"] * 8) +
           "," + ",".join(["1.0"] * 4) +
           ",0,0,0, 5,0,0, 0,3,0, 5,3,2")
    bb = h.iges_bbox(_iges_pd([rec]))
    assert bb["points"] == 4
    assert bb["span_m"]["x"] == pytest.approx(5.0)
    assert bb["span_m"]["y"] == pytest.approx(3.0)
    assert bb["span_m"]["z"] == pytest.approx(2.0)


def test_iges_bbox_rejects_de_pointer_columns():
    """The 8 columns 65-72 hold the directory-entry pointer, not data."""
    text = _iges_pd(["116,0.0,0.0,0.0", "116,1.0,1.0,1.0"])
    assert h.iges_bbox(text)["points"] == 2


def test_t2015_layout_picks_full_scale_column():
    rows = [["Main particulars", "Full scale", "Model 1"], ["Length", "L (m)", "230.0", "7.27"]]
    assert h.t2015_layout(rows) == (1, 2)
    rows2 = [["Main particulars", "Model scale", "Full scale"], ["Draft", "T (m)", "0.112", "5.494"]]
    assert h.t2015_layout(rows2) == (2, 3)
    assert h.table_value(rows2, r"^\s*draft", 3) == pytest.approx(5.494)


def test_cb_from_volume_matches_kcs_publication():
    cb = h.cb_from_volume(52030.0, 230.0, 32.2, 10.8)
    assert h.rel_err(cb, 0.6505) < 1e-4


# --- committed artefacts -------------------------------------------------------
@pytest.fixture(scope="module")
def manifest():
    p = ART / "manifest.json"
    if not p.exists():
        pytest.skip("artefacts not generated")
    return json.loads(p.read_text())


def test_manifest_has_all_four_hulls(manifest):
    names = [s["name"] for s in manifest["ships"]]
    assert set(names) == {"DTC", "ONRT", "JBC", "KCS"}


def test_dtc_geometry_fingerprint_recorded(manifest):
    dtc = next(s for s in manifest["ships"] if s["name"] == "DTC")
    assert dtc["evidence"].startswith("REAL")
    assert "WITHHELD" in dtc["evidence"]              # particulars withheld, not invented
    assert dtc["length_pp"] is None                   # never fabricated
    zips = [u for u in manifest["geometry_files"] if "igs_ordner" in u]
    assert len(zips) == 1
    rec = manifest["geometry_files"][zips[0]]
    assert rec["ok"] and len(rec["sha256"]) == 64
    assert rec["sha256"] == dtc["iges_sha256"]


def test_publication_cb_checks_pass(manifest):
    by_id = {c["id"]: c for c in manifest["checks"]}
    for key in ("JBC-CB-CONSISTENCY", "KCS-CB-CONSISTENCY"):
        assert by_id[key]["verdict"] == "PASS"
        assert by_id[key]["rel_error"] < 0.01


def test_onrt_displacement_dispute_is_resolved_by_density(manifest):
    c = {c["id"]: c for c in manifest["checks"]}["ONRT-DISPLACEMENT-DISPUTE"]
    assert abs(c["implied_density_kg_m3"] - 1000.0) / 1000.0 < 0.01
    assert c["verdict"].startswith("RESOLVED")


def test_geometry_bytes_are_not_committed():
    bad = [p for p in ROOT.rglob("*") if p.suffix.lower() in (".igs", ".iges")
           and ".git" not in p.parts]
    assert bad == []
    assert not list(ART.rglob("*.zip"))


def test_every_geometry_url_has_a_fingerprint(manifest):
    ok = {u: r for u, r in manifest["geometry_files"].items() if r["ok"]}
    assert len(ok) >= 10                              # DTC + ONRT + JBC + KCS archives
    for u, r in ok.items():
        assert len(r["sha256"]) == 64
        assert r["bytes"] > 0
