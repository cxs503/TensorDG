"""Reproducibility guards for the round-2 certified-reference XVAL bundle.

The XVAL document (``docs/XVAL-DG-ROUND2-CERTIFIED-REFERENCE.md``) reports
quantities at the ``0.01 %`` level.  A number that small is only evidence if it
can be **recomputed from committed artifacts**.  These tests enforce the chain

    raw profiles (npz)  ->  verify_xval_round2  ->  xval.json  ->  rendered doc

so that no link in it can drift silently:

* ``test_oss_provenance_is_pinned``      -- the third-party solver is identified
  by sha256 of the exact files used (the first version of the study admitted
  ``clawpack unknown`` / ``missing`` here).
* ``test_certified_reference_reproduces`` -- the certified references, the D1/D2
  gates and every DG leg are recomputed with the *engine* path
  (:mod:`tensordg.certified_reference`), i.e. an implementation independent of
  the one-off driver that produced the claims, and must match.
* ``test_doc_quotes_the_artifacts``      -- every headline number in the document
  is one of the committed numbers (zero hand-transcription).
* ``test_char_decomp_axis_is_closed``    -- the one sensitivity axis the first
  round honestly reported as incomplete is either closed with numbers or the
  document still says so (no silent omission).

Skip policy: if the fixture has not been generated in this checkout
(``artifacts/xval_round2/``), the numeric tests skip with an explicit reason --
they never pass vacuously.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "artifacts" / "xval_round2"
RAW = ROOT / "validation" / "repro_shu_osher" / "artifacts" / "raw"
DOC = ROOT / "docs" / "XVAL-DG-ROUND2-CERTIFIED-REFERENCE.md"
MANIFEST = OUTDIR / "oss_provenance.json"

needs_fixture = pytest.mark.skipif(
    not (OUTDIR / "xval.json").exists(),
    reason="artifacts/xval_round2/xval.json not generated in this checkout "
           "(run benchmarks.xval_oss_shu_osher --warm + tools.verify_xval_round2)")


def _xval() -> dict:
    return json.loads((OUTDIR / "xval.json").read_text())


def _verify() -> dict:
    """Run the second-path reproduction once per session (cheap: pure numpy)."""
    from tools import verify_xval_round2

    return verify_xval_round2.reproduce(OUTDIR, RAW, OUTDIR / "xval.json")


# ------------------------------------------------------------------ provenance
@pytest.mark.skipif(not MANIFEST.exists(),
                    reason="artifacts/xval_round2/oss_provenance.json not generated")
def test_oss_provenance_is_pinned():
    from tools.fetch_oss_clawpack import CLAWPACK

    prov = json.loads(MANIFEST.read_text())
    assert prov["version"] == CLAWPACK["version"] == "5.14.0"
    assert prov["tarball"]["sha256"] == CLAWPACK["sha256"]
    assert prov["tarball"]["bytes"] == CLAWPACK["bytes"]
    # the Riemann solver actually executed must be identified by its bytes
    euler = prov["study_files"]["riemann/riemann/euler_1D_py.py"]
    assert "missing" not in euler and len(euler["sha256"]) == 64
    # the hollowness is *declared*, not hidden: placeholders are logged
    assert isinstance(prov["placeholders"], list)
    # the pure-Python 1D kernel may never be one of the hollowed-out modules
    assert not any("euler_1D_py" in p for p in prov["placeholders"])
    # provenance recorded by the run itself must agree with the manifest
    if not (OUTDIR / "xval.json").exists():
        pytest.skip("manifest pinned; cross-check against a live run needs xval.json")
    tree = _xval()["oss_tree"]
    assert tree["version"] == "5.14.0"
    assert tree["files"]["clawpack/riemann/euler_1D_py.py"]["sha256"] == euler["sha256"]


# ------------------------------------------------------------------ reproduction
@needs_fixture
def test_certified_reference_reproduces():
    result = _verify()
    assert result["status"].startswith("REPRODUCED"), (
        f"claims not reproducible: {result['mismatches']}")
    rec = result["recomputed"]
    # both ladders must certify on their own terms, not just agree
    assert rec["ours"]["verdict"] == "CERTIFIED"
    assert rec["oss"]["verdict"] == "CERTIFIED"
    # pre-declared gates
    assert rec["d1_percent"] < 1.0
    assert rec["d2_passes"]
    # D3: every DG leg still inside the 5 % gate under the OSS reference
    assert all(leg.get("passes_5pct") for leg in rec["dg_legs"] if not leg.get("missing"))
    # the reference band must not consume the gate (otherwise the comparison is vacuous)
    assert rec["oss"]["band_percent"] < 5.0 / 3.0


@needs_fixture
def test_engine_and_driver_agree_on_bands():
    """Two implementations of the same statistics must agree at round-off."""
    rec = _verify()["recomputed"]
    study = _xval()
    assert rec["ours"]["p_obs"] == pytest.approx(study["ours"]["observed_order_used"], rel=1e-9)
    assert rec["oss"]["p_obs"] == pytest.approx(study["oss"]["observed_order_used"], rel=1e-9)
    assert rec["ours"]["band_percent"] == pytest.approx(
        study["ours"]["gci_finest_percent"], rel=1e-9)
    assert rec["oss"]["band_percent"] == pytest.approx(
        study["oss"]["gci_finest_percent"], rel=1e-9)


# ------------------------------------------------------------------ doc binding
@needs_fixture
def test_doc_quotes_the_artifacts():
    doc = DOC.read_text()
    study = _xval()
    cert = study["certified"]
    # headline numbers, formatted exactly as the renderer writes them
    for value, digits in ((cert["difference_percent"], 4),
                          (cert["combined_band_percent"], 4),
                          (study["ours"]["gci_finest_percent"], 6),
                          (study["oss"]["gci_finest_percent"], 6),
                          (study["ours"]["observed_order_used"], 4),
                          (study["oss"]["observed_order_used"], 4)):
        assert f"{value:.{digits}f}" in doc, f"{value:.{digits}f} missing from the document"
    # the evidence path must name the committed fixture, not a scratch directory
    assert "artifacts/xval_round2" in doc
    assert "/tmp/oss" not in doc


# ------------------------------------------------------------------ honest gap
@needs_fixture
def test_char_decomp_axis_is_closed():
    study = _xval()
    axis = study["sensitivity"]["char_decomp"]
    doc = DOC.read_text()
    if axis.get("incomplete"):
        assert "char_decomp" in doc and "未完成" in doc, \
            "incomplete axis silently dropped from the document"
    else:
        assert axis["certified_ref_diff_percent"] < 1.0
        assert "char_decomp" in doc
        rec = _verify()["recomputed"]["char_decomp"]
        assert rec is not None
        assert rec["certified_ref_diff_percent"] == pytest.approx(
            axis["certified_ref_diff_percent"], rel=1e-9)